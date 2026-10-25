"""Tests for the FIP music provider."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.enums import ContentType, EventType, MediaType, StreamType
from music_assistant_models.errors import MediaNotFoundError, UnplayableMediaError
from music_assistant_models.event import MassEvent

from music_assistant.helpers.throttle_retry import Throttler
from music_assistant.providers.fip import SUPPORTED_FEATURES
from music_assistant.providers.fip.constants import (
    FIP_STATIONS,
    OVERDUE_UPDATE_INTERVAL,
    STREAM_DELAY,
    STREAM_METADATA_UPDATE_INTERVAL,
)
from music_assistant.providers.fip.helpers import LiveDisplay, LiveItem, parse_radio
from music_assistant.providers.fip.provider import LIVE_DISPLAY_KEY, FipProvider

if TYPE_CHECKING:
    from music_assistant_models.media_items import Radio

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> dict[str, Any]:
    """Return a recorded answer of the now-playing service."""
    data: dict[str, Any] = json.loads((FIXTURES / f"{name}.json").read_text())
    return data


def _song_now(start: float, length: int = 200) -> dict[str, Any]:
    """Return an answer whose current song started at the given time."""
    return {
        "now": {
            "firstLine": "Song",
            "secondLine": "Artist",
            "cover": "cover",
            "startTime": int(start),
            "endTime": int(start) + length,
            "songUuid": "song",
        },
        "next": [],
    }


@pytest.fixture
def provider() -> FipProvider:
    """Return a FipProvider wired to mocks for unit testing."""
    manifest = MagicMock()
    manifest.domain = "fip"
    manifest.name = "FIP"

    config = MagicMock()
    config.instance_id = "fip_test"
    config.values = {}

    def _get_value(key: str, default: Any = None) -> Any:
        if key == "log_level":
            return "GLOBAL"
        return default

    config.get_value.side_effect = _get_value
    provider = FipProvider(MagicMock(), manifest, config, SUPPORTED_FEATURES)
    provider._station_displays = {station_id: LiveDisplay() for station_id in FIP_STATIONS}
    provider._unsubscribe_library = MagicMock()
    provider._get_pull = AsyncMock(return_value=None)  # type: ignore[method-assign]
    return provider


async def test_get_radio_and_unknown_station(provider: FipProvider) -> None:
    """A known station resolves, an unknown one raises."""
    provider._get_live = AsyncMock(return_value=None)  # type: ignore[method-assign]
    radio = await provider.get_radio("fip_jazz")
    assert radio.name == "FIP Jazz"
    assert radio.metadata.description == FIP_STATIONS["fip_jazz"]["description"]
    with pytest.raises(MediaNotFoundError):
        await provider.get_radio("france_inter")


async def test_a_station_asked_for_shows_what_plays_and_follows_the_music(
    provider: FipProvider,
) -> None:
    """A station looked up on its own is live, and is looked at again when its song ends."""
    start = time.time() - 30
    provider._get_live = AsyncMock(return_value=_song_now(start))  # type: ignore[method-assign]
    radio = await provider.get_radio("fip_jazz")
    assert radio.metadata.description is not None
    assert radio.metadata.description.startswith("Song — Artist\n\n")
    call_later = cast("MagicMock", provider.mass.call_later)
    _, target, station_id = call_later.call_args.args
    assert (target, station_id) == (provider._update_station, "fip_jazz")


def _library_entry(provider: FipProvider, live: bool, instance_id: str | None = None) -> Radio:
    """Return a library entry made from FIP Jazz, as it was when it was on air or not."""
    station = parse_radio(
        "fip_jazz",
        instance_id or provider.instance_id,
        "fip",
        LiveItem("Song", "Artist", "cover", 1000, 1200, "song") if live else None,
    )
    station.provider = "library"
    station.item_id = "3"
    return station


async def test_a_library_entry_keeps_the_station_picture_and_text(provider: FipProvider) -> None:
    """A library entry made while a song was on air gets the station's own back."""
    update = AsyncMock()
    cast("MagicMock", provider.mass.music.radio).update_item_in_library = update
    entry = _library_entry(provider, live=True)

    await provider._keep_library_entry_plain(
        MassEvent(EventType.MEDIA_ITEM_ADDED, entry.uri, entry)
    )

    item_id, fixed = update.call_args.args
    assert item_id == "3"
    assert update.call_args.kwargs == {"overwrite": True}
    station = parse_radio("fip_jazz", provider.instance_id, "fip")
    assert fixed.metadata.description == station.metadata.description
    assert fixed.metadata.images == station.metadata.images
    assert fixed.provider_mappings == entry.provider_mappings


async def test_a_plain_or_foreign_library_entry_is_left_alone(provider: FipProvider) -> None:
    """Only a library entry of this provider showing a song is changed."""
    update = AsyncMock()
    cast("MagicMock", provider.mass.music.radio).update_item_in_library = update
    for entry in (
        _library_entry(provider, live=False),
        _library_entry(provider, live=True, instance_id="another_fip"),
    ):
        await provider._keep_library_entry_plain(
            MassEvent(EventType.MEDIA_ITEM_UPDATED, entry.uri, entry)
        )
    update.assert_not_called()


async def test_search_matches_station_names(provider: FipProvider) -> None:
    """Search finds stations by name, for radio only."""
    results = await provider.search("metal", [MediaType.RADIO])
    assert [radio.item_id for radio in results.radio] == ["fip_metal"]
    assert (await provider.search("fip", [MediaType.RADIO], limit=3)).radio
    assert len((await provider.search("fip", [MediaType.RADIO], limit=3)).radio) == 3
    assert not (await provider.search("metal", [MediaType.TRACK])).radio


async def test_browse_lists_all_thirteen_stations(provider: FipProvider) -> None:
    """Browsing lists FIP and its twelve webradios, each with what it plays now."""
    start = time.time() - 30
    provider._get_live = AsyncMock(return_value=_song_now(start))  # type: ignore[method-assign]
    items = await provider.browse("fip://")
    assert len(items) == 13
    assert items[0].name == "FIP"
    radio = cast("Radio", items[0])
    assert radio.metadata.description is not None
    assert radio.metadata.description.startswith("Song — Artist\n\n")
    assert radio.metadata.images
    assert radio.metadata.images[0].path.endswith("/cover/400x400")


async def test_a_station_on_the_placeholder_shows_the_older_services_song(
    provider: FipProvider,
) -> None:
    """While the service answers with its placeholder, the older service's song stands in."""
    provider._get_live = AsyncMock(return_value=_fixture("webradio_placeholder"))  # type: ignore[method-assign]
    provider._get_pull = AsyncMock(return_value=_fixture("rock_pull"))  # type: ignore[method-assign]
    with patch("music_assistant.providers.fip.provider.time.time", return_value=1792838425):
        radio = await provider.get_radio("fip_rock")
    assert radio.metadata.description is not None
    assert radio.metadata.description.startswith("Rise — Public Image Limited\n\n")


async def test_browse_keeps_each_station_up_to_date(provider: FipProvider) -> None:
    """Each station is looked at again when its song ends, and a new song is sent out."""
    start = time.time() - 30
    provider._get_live = AsyncMock(return_value=_song_now(start))  # type: ignore[method-assign]
    await provider.browse("fip://")
    call_later = cast("MagicMock", provider.mass.call_later)
    assert call_later.call_count == 13
    delay, target, station_id = call_later.call_args_list[0].args
    assert target == provider._update_station
    assert station_id == "fip"
    assert delay == pytest.approx(start + 200 + STREAM_DELAY - time.time(), abs=2)

    signal_event = cast("MagicMock", provider.mass.signal_event)
    await provider._update_station("fip")
    signal_event.assert_not_called()
    provider._get_live.return_value = _song_now(time.time())
    await provider._update_station("fip")
    event, uri, radio = signal_event.call_args.args
    assert event == EventType.MEDIA_ITEM_UPDATED
    assert uri == f"{provider.instance_id}://radio/fip"
    assert radio.uri == uri


async def test_the_list_is_left_alone_once_nobody_browses(provider: FipProvider) -> None:
    """After the watch period a station is no longer looked at again."""
    provider._get_live = AsyncMock(return_value=_song_now(time.time()))  # type: ignore[method-assign]
    provider._watch_until = time.time() - 1
    await provider._update_station("fip_jazz")
    cast("MagicMock", provider.mass.call_later).assert_not_called()


async def test_unload_stops_the_station_timers(provider: FipProvider) -> None:
    """Unloading the provider cancels every station's timer."""
    with patch("music_assistant.models.provider.Provider.unload", AsyncMock()):
        await provider.unload()
    assert cast("MagicMock", provider.mass.cancel_timer).call_count == 13


async def test_stream_details_wire_the_live_metadata(provider: FipProvider) -> None:
    """The stream plays the station's Icecast mount and starts with the live song."""
    provider._get_live = AsyncMock(  # type: ignore[method-assign]
        return_value=_song_now(time.time())
    )
    details = await provider.get_stream_details("fip_sacre_francais", MediaType.RADIO)

    assert details.path == "https://icecast.radiofrance.fr/fipsacrefrancais-hifi.aac"
    assert details.stream_type == StreamType.HTTP
    assert details.audio_format.content_type == ContentType.AAC
    assert details.audio_format.sample_rate == 48000
    assert details.stream_metadata_update_callback == provider._update_stream_metadata
    assert details.stream_metadata_update_interval == STREAM_METADATA_UPDATE_INTERVAL
    assert details.stream_metadata is not None
    assert details.stream_metadata.title == "Song"
    provider._get_live.assert_awaited_once_with(96)


async def test_get_live_reads_the_service(provider: FipProvider) -> None:
    """The live answer comes from the service, and a failed request gives None."""
    # the method itself, without the cache in front of it
    get_live = cast("Any", FipProvider._get_live).__wrapped__
    provider._throttler = Throttler(rate_limit=5, period=1.0)
    response = MagicMock(status=200)
    response.json = AsyncMock(return_value=_fixture("webradio_song"))
    request = MagicMock()
    request.__aenter__ = AsyncMock(return_value=response)
    request.__aexit__ = AsyncMock(return_value=None)
    session = cast("MagicMock", provider.mass.http_session)
    session.get.return_value = request

    data = await get_live(provider, 65)
    assert data == _fixture("webradio_song")
    url = session.get.call_args.args[0]
    assert url == "https://api.radiofrance.fr/livemeta/live/65/webrf_webradio_player"

    response.status = 503
    assert await get_live(provider, 66) is None
    request.__aenter__.side_effect = TimeoutError
    assert await get_live(provider, 64) is None


async def test_stream_details_refuse_other_media(provider: FipProvider) -> None:
    """Only radio items of this provider can be streamed."""
    with pytest.raises(UnplayableMediaError):
        await provider.get_stream_details("fip", MediaType.TRACK)
    with pytest.raises(MediaNotFoundError):
        await provider.get_stream_details("france_inter", MediaType.RADIO)


async def test_update_asks_more_often_while_overdue(provider: FipProvider) -> None:
    """The callback shortens its interval once the song should be over, and restores it."""
    provider._get_live = AsyncMock(  # type: ignore[method-assign]
        return_value=_song_now(time.time() - 300)
    )
    details = await provider.get_stream_details("fip", MediaType.RADIO)
    assert details.stream_metadata_update_interval == OVERDUE_UPDATE_INTERVAL

    provider._get_live.return_value = _song_now(time.time())
    await provider._update_stream_metadata(details, 0)
    assert details.stream_metadata_update_interval == STREAM_METADATA_UPDATE_INTERVAL


async def test_a_failed_request_keeps_the_display(provider: FipProvider) -> None:
    """When the service does not answer, the metadata on display stays."""
    provider._get_live = AsyncMock(  # type: ignore[method-assign]
        return_value=_fixture("webradio_song")
    )
    details = await provider.get_stream_details("fip_jazz", MediaType.RADIO)
    shown = details.stream_metadata

    provider._get_live.return_value = None
    await provider._update_stream_metadata(details, 0)
    assert details.stream_metadata is shown
    assert details.data[LIVE_DISPLAY_KEY].shown is not None


async def test_a_placeholder_shows_the_station(provider: FipProvider) -> None:
    """A station on its placeholder gives no metadata, so the player shows the station."""
    provider._get_live = AsyncMock(  # type: ignore[method-assign]
        return_value=_fixture("webradio_placeholder")
    )
    details = await provider.get_stream_details("fip_groove", MediaType.RADIO)
    assert details.stream_metadata is None
