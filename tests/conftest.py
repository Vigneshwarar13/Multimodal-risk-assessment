"""Pytest configuration and fixtures for the test suite."""
import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_audio_path():
    """Return path to sample audio file."""
    return Path(__file__).parent / "data" / "sample_audio.wav"


@pytest.fixture
def sample_frame():
    """Return a random 48x48 grayscale image as numpy array."""
    return np.random.randint(0, 256, (48, 48), dtype=np.uint8)


@pytest.fixture
def sample_face_tensor():
    """Return normalized face tensor (1, 48, 48, 1)."""
    face = np.random.randint(0, 256, (48, 48), dtype=np.uint8).astype(np.float32) / 255.0
    return face.reshape(1, 48, 48, 1)


@pytest.fixture
def sample_tamil_text():
    """Return sample Tamil text for testing."""
    return "இந்த ஒப்பந்தம் கையெழுத்திட வேண்டும்"


@pytest.fixture
def sample_tamil_coercion_text():
    """Return sample Tamil coercion text."""
    return "நீங்கள் கையெழுத்திட வேண்டும் அல்லது தசக்ஸ் கொடுக்க வேண்டும்"
