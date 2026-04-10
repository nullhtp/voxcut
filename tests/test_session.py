"""Tests for session persistence."""
from pathlib import Path

from sam3_audio.fragment import Fragment
from sam3_audio.session import Session, load, save, sidecar_path


def test_sidecar_path():
    assert sidecar_path(Path("foo.mp3")) == Path("foo.mp3.sam3.json")


def test_roundtrip(tmp_path: Path):
    audio = tmp_path / "test.mp3"
    audio.touch()
    sess = Session(
        fragments=[Fragment(1.0, 2.0), Fragment(3.0, 5.0)],
        last_description="man speaking",
        description_history=["man speaking", "woman singing"],
    )
    save(audio, sess)
    loaded = load(audio)
    assert len(loaded.fragments) == 2
    assert loaded.fragments[0].start == 1.0
    assert loaded.fragments[1].end == 5.0
    assert loaded.last_description == "man speaking"
    assert loaded.description_history == ["man speaking", "woman singing"]


def test_empty_session_removes_sidecar(tmp_path: Path):
    audio = tmp_path / "test.mp3"
    audio.touch()
    # Save non-empty, then empty
    save(audio, Session(fragments=[Fragment(1.0, 2.0)]))
    assert sidecar_path(audio).exists()
    save(audio, Session())
    assert not sidecar_path(audio).exists()


def test_load_missing_file(tmp_path: Path):
    sess = load(tmp_path / "nonexistent.mp3")
    assert sess.fragments == []
    assert sess.last_description == ""


def test_load_corrupt_json(tmp_path: Path):
    audio = tmp_path / "test.mp3"
    audio.touch()
    sidecar_path(audio).write_text("{bad json")
    sess = load(audio)
    assert sess.fragments == []


def test_add_description():
    sess = Session()
    sess.add_description("foo")
    sess.add_description("bar")
    sess.add_description("foo")
    assert sess.description_history == ["foo", "bar"]
    assert sess.last_description == "foo"
