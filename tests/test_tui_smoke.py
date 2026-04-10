"""Smoke tests for the TUI application."""
from __future__ import annotations

from pathlib import Path

import pytest

from sam3_audio.separator import SamSeparator
from sam3_audio.tui import AudioTUI


@pytest.fixture
def make_app(wav_10s: Path):
    """Create an AudioTUI instance for testing."""
    def _make(**kwargs):
        return AudioTUI(wav_10s, wav_10s, SamSeparator(), **kwargs)
    return _make


async def test_boots(make_app):
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.src is not None


async def test_mark_in_out(make_app):
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.player.seek_to(2.0)
        await pilot.press("i")
        app.player.seek_to(5.0)
        await pilot.press("o")
        await pilot.pause()
        assert len(app.fragments) == 1
        assert app.fragments[0].start == 2.0
        assert app.fragments[0].end == 5.0


async def test_delete_and_undo(make_app):
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.player.seek_to(1.0)
        await pilot.press("i")
        app.player.seek_to(3.0)
        await pilot.press("o")
        await pilot.pause()
        assert len(app.fragments) == 1
        await pilot.press("x")
        await pilot.pause()
        assert len(app.fragments) == 0
        await pilot.press("u")
        await pilot.pause()
        assert len(app.fragments) == 1


async def test_split_and_merge(make_app):
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.player.seek_to(2.0)
        await pilot.press("i")
        app.player.seek_to(8.0)
        await pilot.press("o")
        await pilot.pause()
        # Split at 5.0
        app.player.seek_to(5.0)
        await pilot.press("S")
        await pilot.pause()
        assert len(app.fragments) == 2
        # Merge
        await pilot.press("m")
        await pilot.pause()
        assert len(app.fragments) == 1
        assert app.fragments[0].start == 2.0
        assert app.fragments[0].end == 8.0


async def test_help_screen(make_app):
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("question_mark")
        await pilot.pause()
        assert len(app.screen_stack) > 1
        await pilot.press("escape")
        await pilot.pause()
        assert len(app.screen_stack) == 1


async def test_dirty_flag(make_app):
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not app._dirty
        app.player.seek_to(1.0)
        await pilot.press("i")
        app.player.seek_to(2.0)
        await pilot.press("o")
        await pilot.pause()
        assert app._dirty


async def test_no_file_launch():
    app = AudioTUI(None, None, SamSeparator())
    async with app.run_test() as pilot:
        await pilot.pause()
        # File picker should be pushed; dismiss it
        await pilot.press("escape")
        await pilot.pause()


async def test_gain(make_app):
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("v")
        await pilot.pause()
        assert app.player.gain_db == 3.0
        await pilot.press("V")
        await pilot.pause()
        assert app.player.gain_db == 0.0
