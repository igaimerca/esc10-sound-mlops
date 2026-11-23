import io

import numpy as np
import pytest
import soundfile as sf

from src.preprocessing import INPUT_SHAPE, SR, fix_length, load_audio, log_mel


def wav_bytes(seconds=2.0, sr=44100):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    buf = io.BytesIO()
    sf.write(buf, 0.3 * np.sin(2 * np.pi * 440 * t), sr, format="WAV")
    return buf.getvalue()


def test_load_audio_resamples_and_pads_to_five_seconds():
    y = load_audio(io.BytesIO(wav_bytes()))
    assert y.shape == (5 * SR,)
    assert np.abs(y[-SR:]).max() == 0


def test_long_audio_is_cropped():
    assert fix_length(np.ones(10 * SR)).shape == (5 * SR,)


def test_log_mel_shape_and_standardised():
    S = log_mel(load_audio(io.BytesIO(wav_bytes())))
    assert S.shape == INPUT_SHAPE
    assert abs(S.mean()) < 1e-3 and abs(S.std() - 1) < 1e-2


def test_garbage_bytes_rejected():
    with pytest.raises(ValueError):
        load_audio(io.BytesIO(b"not audio"))


def test_silent_clip_does_not_produce_nan():
    S = log_mel(np.zeros(5 * SR, np.float32))
    assert np.isfinite(S).all()
