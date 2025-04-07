from core.stream_manager import StreamManager
from unittest.mock import MagicMock


def test_stream_manager_pause_resume():
    sm = StreamManager()
    sm.stream = MagicMock()
    sm.stream.frames = MagicMock(return_value=iter([b"frame1", b"frame2"]))

    sm.pause()
    assert sm.is_streaming is False

    sm.resume()
    assert sm.is_streaming is True


def test_stream_manager_generate_pauses():
    sm = StreamManager()
    sm.stream = MagicMock()
    sm.stream.frames = MagicMock(return_value=iter([b"frame1"]))

    sm.pause()
    gen = sm.generate()

    # Avance un peu dans le generateur, il ne doit rien yield
    try:
        next(gen)
        assert False, "Le stream ne doit pas yield lorsqu'il est en pause"
    except StopIteration:
        pass
