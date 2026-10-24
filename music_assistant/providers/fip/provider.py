"""FIP Music Provider for Music Assistant."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Any

import aiohttp
from music_assistant_models.enums import ContentType, EventType, MediaType, StreamType
from music_assistant_models.errors import MediaNotFoundError, UnplayableMediaError
from music_assistant_models.media_items import (
    AudioFormat,
    BrowseFolder,
    ItemMapping,
    MediaItemType,
    Radio,
    SearchResults,
)
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.controllers.cache import use_cache
from music_assistant.helpers.throttle_retry import Throttler
from music_assistant.models.music_provider import MusicProvider

from .constants import (
    API_TIMEOUT,
    FIP_STATIONS,
    LIVE_CACHE_EXPIRATION,
    LIVE_URL,
    OVERDUE_UPDATE_INTERVAL,
    STATION_LIST_WATCH,
    STREAM_METADATA_UPDATE_INTERVAL,
    STREAM_URL,
)
from .helpers import LiveDisplay, build_stream_metadata, parse_live, parse_radio

# Key in StreamDetails.data under which a playing stream keeps its LiveDisplay.
LIVE_DISPLAY_KEY = "fip_live_display"


class FipProvider(MusicProvider):
    """FIP Music Provider for Music Assistant."""

    _throttler: Throttler
    # what each station shows in the station list
    _station_displays: dict[str, LiveDisplay]
    # until when the station list is kept up to date (see STATION_LIST_WATCH)
    _watch_until: float = 0.0

    @property
    def max_concurrent_streams(self) -> None:
        """Allow unlimited concurrent upstream source streams."""
        return None

    async def handle_async_init(self) -> None:
        """Handle async initialization of the provider."""
        self._throttler = Throttler(rate_limit=5, period=1.0)
        self._station_displays = {station_id: LiveDisplay() for station_id in FIP_STATIONS}

    async def unload(self, is_removed: bool = False) -> None:
        """Handle unload/close of the provider."""
        for station_id in FIP_STATIONS:
            self.mass.cancel_timer(self._station_task_id(station_id))
            self.mass.cancel_task(self._station_task_id(station_id))
        await super().unload(is_removed)

    async def get_radio(self, prov_radio_id: str) -> Radio:
        """
        Get a station, showing what it plays now.

        Like a browsed station, it is then kept up to date for a while (see STATION_LIST_WATCH),
        so an open details view follows the music.
        """
        if prov_radio_id not in FIP_STATIONS:
            raise MediaNotFoundError(f"Unknown radio station: {prov_radio_id}")
        self._watch_until = max(self._watch_until, time.time() + STATION_LIST_WATCH)
        await self._update_station(prov_radio_id, notify=False)
        return self._station_radio(prov_radio_id)

    async def search(
        self,
        search_query: str,
        media_types: list[MediaType],
        limit: int = 5,
    ) -> SearchResults:
        """Perform search on the FIP stations."""
        results = SearchResults()
        if MediaType.RADIO not in media_types:
            return results
        search_query_lower = search_query.lower().strip()
        if not search_query_lower:
            return results
        radios: list[Radio] = []
        for station_id, station in FIP_STATIONS.items():
            if search_query_lower in station["name"].lower():
                radios.append(parse_radio(station_id, self.instance_id, self.domain))
                if len(radios) >= limit:
                    break
        results.radio = radios
        return results

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """Get streamdetails for a radio station."""
        if media_type != MediaType.RADIO:
            raise UnplayableMediaError(f"Unsupported media type: {media_type}")
        if item_id not in FIP_STATIONS:
            raise MediaNotFoundError(f"Unknown radio station: {item_id}")

        stream_details = StreamDetails(
            item_id=item_id,
            provider=self.instance_id,
            audio_format=AudioFormat(
                content_type=ContentType.AAC,
                channels=2,
            ),
            media_type=MediaType.RADIO,
            stream_type=StreamType.HTTP,
            path=STREAM_URL.format(slug=item_id.replace("_", "")),
            allow_seek=False,
            can_seek=False,
            duration=0,
            stream_metadata_update_callback=self._update_stream_metadata,
            stream_metadata_update_interval=STREAM_METADATA_UPDATE_INTERVAL,
            data={LIVE_DISPLAY_KEY: LiveDisplay()},
        )
        # Set initial metadata so the first frame the listener sees
        # is the live song rather than an empty banner.
        await self._update_stream_metadata(stream_details, 0)
        return stream_details

    async def browse(self, path: str) -> Sequence[MediaItemType | ItemMapping | BrowseFolder]:
        """
        Browse this provider's items: the stations, each showing what it plays now.

        From then on, for a while (see STATION_LIST_WATCH), each station is asked again when
        its song ends, and an open browse view is sent the station's new entry.
        """
        self._watch_until = time.time() + STATION_LIST_WATCH
        await asyncio.gather(
            *(self._update_station(station_id, notify=False) for station_id in FIP_STATIONS)
        )
        return [self._station_radio(station_id) for station_id in FIP_STATIONS]

    def _station_radio(self, station_id: str) -> Radio:
        """Return a station's entry in the station list, with what it plays now."""
        live = self._station_displays[station_id].shown
        return parse_radio(station_id, self.instance_id, self.domain, live)

    def _station_task_id(self, station_id: str) -> str:
        """Return the id of the timer that keeps a station's list entry up to date."""
        return f"fip_station_{self.instance_id}_{station_id}"

    async def _update_station(self, station_id: str, notify: bool = True) -> None:
        """
        Bring a station's list entry up to date, and look again when it is due.

        :param station_id: A known FIP station id.
        :param notify: Whether to send the new entry to open browse views when it changed.
        """
        display = self._station_displays[station_id]
        data = await self._get_live(FIP_STATIONS[station_id]["api_id"])
        now = time.time()
        if data is not None and display.update(*parse_live(data), now) and notify:
            radio = self._station_radio(station_id)
            self.mass.signal_event(EventType.MEDIA_ITEM_UPDATED, radio.uri, radio)
        if now < self._watch_until:
            self.mass.call_later(
                display.next_look(now),
                self._update_station,
                station_id,
                task_id=self._station_task_id(station_id),
            )

    @use_cache(LIVE_CACHE_EXPIRATION, cache_none=False)
    async def _get_live(self, api_id: int) -> dict[str, Any] | None:
        """
        Get the now-playing answer for a station, or None if unavailable.

        :param api_id: The station's id at the now-playing service.
        """
        url = LIVE_URL.format(api_id=api_id)
        try:
            async with (
                self._throttler,
                self.mass.http_session.get(url, timeout=API_TIMEOUT) as response,
            ):
                if response.status != 200:
                    self.logger.debug(
                        "FIP now-playing service returned status %s for station %s",
                        response.status,
                        api_id,
                    )
                    return None
                data: dict[str, Any] = await response.json()
        except (TimeoutError, aiohttp.ClientError, ValueError) as err:
            self.logger.debug("FIP now-playing request failed for station %s: %s", api_id, err)
            return None
        return data

    async def _update_stream_metadata(
        self, stream_details: StreamDetails, elapsed_time: int
    ) -> None:
        """
        Update stream metadata callback called by player queue controller.

        :param stream_details: StreamDetails object to update with metadata.
        :param elapsed_time: Elapsed playback time in seconds (unused for FIP).
        """
        display: LiveDisplay = stream_details.data[LIVE_DISPLAY_KEY]
        data = await self._get_live(FIP_STATIONS[stream_details.item_id]["api_id"])
        now = time.time()
        if data is not None:
            current, upcoming = parse_live(data)
            if display.update(current, upcoming, now):
                stream_details.stream_metadata = build_stream_metadata(display.shown, now)
                self.logger.debug(
                    "Updating stream metadata for %s: %s",
                    stream_details.item_id,
                    display.shown.title if display.shown else None,
                )
        # ask more often while the song on display should already be over
        stream_details.stream_metadata_update_interval = (
            OVERDUE_UPDATE_INTERVAL if display.overdue(now) else STREAM_METADATA_UPDATE_INTERVAL
        )
