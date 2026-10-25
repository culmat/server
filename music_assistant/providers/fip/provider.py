"""FIP Music Provider for Music Assistant."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

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
    PULL_URL,
    STATION_LIST_WATCH,
    STREAM_METADATA_UPDATE_INTERVAL,
    STREAM_URL,
)
from .helpers import (
    LiveDisplay,
    LiveItem,
    build_stream_metadata,
    parse_live,
    parse_pull,
    parse_radio,
)

if TYPE_CHECKING:
    from music_assistant_models.event import MassEvent

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
        self._unsubscribe_library = self.mass.subscribe(
            self._keep_library_entry_plain,
            (EventType.MEDIA_ITEM_ADDED, EventType.MEDIA_ITEM_UPDATED),
        )

    async def unload(self, is_removed: bool = False) -> None:
        """Handle unload/close of the provider."""
        self._unsubscribe_library()
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
        # announced as well: a station list shown from cache learns of it only that way
        await self._update_station(prov_radio_id)
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
            # Every hifi mount is 48 kHz. Left out, the format says 44.1 kHz, and the
            # stream is resampled down when decoded and up again for a 48 kHz player.
            audio_format=AudioFormat(
                content_type=ContentType.AAC,
                sample_rate=48000,
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

    async def _keep_library_entry_plain(self, event: MassEvent) -> None:
        """
        Give a library entry made from a station the station's own picture and description.

        A station shows what it plays now, and a library keeps what it is given, so an entry
        would otherwise keep a song long over.

        :param event: A library or provider item that was added or updated.
        """
        item = event.data
        if not isinstance(item, Radio) or item.provider != "library":
            return
        station_id = next(
            (
                mapping.item_id
                for mapping in item.provider_mappings
                if mapping.provider_instance == self.instance_id and mapping.item_id in FIP_STATIONS
            ),
            None,
        )
        if station_id is None:
            return
        station = parse_radio(station_id, self.instance_id, self.domain)
        if (
            item.metadata.description == station.metadata.description
            and item.metadata.images == station.metadata.images
        ):
            return
        item.metadata.description = station.metadata.description
        item.metadata.images = station.metadata.images
        await self.mass.music.radio.update_item_in_library(item.item_id, item, overwrite=True)

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
        :param notify: Whether to send the new entry to open views when it changed.
        """
        display = self._station_displays[station_id]
        answer = await self._live_answer(FIP_STATIONS[station_id]["api_id"])
        now = time.time()
        if answer is not None and display.update(*answer, now) and notify:
            radio = self._station_radio(station_id)
            self.mass.signal_event(EventType.MEDIA_ITEM_UPDATED, radio.uri, radio)
        if now < self._watch_until:
            self.mass.call_later(
                display.next_look(now),
                self._update_station,
                station_id,
                task_id=self._station_task_id(station_id),
            )

    async def _live_answer(self, api_id: int) -> tuple[LiveItem | None, LiveItem | None] | None:
        """
        Get what a station plays now and next, or None if the service is unavailable.

        While the service answers with its placeholder, the older service stands in when it
        knows the song on air.

        :param api_id: The station's id at the now-playing service.
        """
        data = await self._get_live(api_id)
        if data is None:
            return None
        current, upcoming = parse_live(data)
        if current is None or not current.is_timed:
            pulled = await self._get_pull(api_id)
            if pulled is not None and (song := parse_pull(pulled, time.time()))[0] is not None:
                return song
        return current, upcoming

    @use_cache(LIVE_CACHE_EXPIRATION, cache_none=False)
    async def _get_live(self, api_id: int) -> dict[str, Any] | None:
        """
        Get the now-playing answer for a station, or None if unavailable.

        :param api_id: The station's id at the now-playing service.
        """
        return await self._fetch_json(LIVE_URL.format(api_id=api_id), api_id)

    @use_cache(LIVE_CACHE_EXPIRATION, cache_none=False)
    async def _get_pull(self, api_id: int) -> dict[str, Any] | None:
        """
        Get the older now-playing service's answer for a station, or None if unavailable.

        :param api_id: The station's id at the now-playing service.
        """
        return await self._fetch_json(PULL_URL.format(api_id=api_id), api_id)

    async def _fetch_json(self, url: str, api_id: int) -> dict[str, Any] | None:
        """
        Get a JSON answer of a now-playing service, or None if unavailable.

        :param url: The address to ask.
        :param api_id: The station's id, for the log.
        """
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
        answer = await self._live_answer(FIP_STATIONS[stream_details.item_id]["api_id"])
        now = time.time()
        if answer is not None:
            if display.update(*answer, now):
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
