import pytest
import respx
import httpx
from pathlib import Path
from audio_api.audio_downloader import download_audio_file

TEST_URL = "https://example.com/test.wav"
TEST_BYTES = b"RIFF" + b"\x00" * 1024  # fake WAV-like data

@respx.mock
@pytest.mark.asyncio
async def test_download_success():
    respx.get(TEST_URL).respond(200, content=TEST_BYTES)

    downloaded_path = await download_audio_file(TEST_URL)

    assert isinstance(downloaded_path, Path)
    assert downloaded_path.exists()
    assert downloaded_path.read_bytes() == TEST_BYTES
    assert downloaded_path.name.endswith(".wav")

    downloaded_path.unlink()


@respx.mock
@pytest.mark.asyncio
async def test_download_fails_on_http_error():
    respx.get(TEST_URL).respond(404)
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        await download_audio_file(TEST_URL)

    assert excinfo.value.response.status_code == 404

@pytest.mark.asyncio
async def test_download_fails_on_invalid_url_scheme():
    """
    Tests that the function raises a ValueError for non-http/https URLs.
    """
    invalid_url = "ftp://example.com/unsupported.mp3"

    with pytest.raises(ValueError):
        await download_audio_file(invalid_url)


@respx.mock
@pytest.mark.asyncio
async def test_download_handles_redirects():
    """
    Tests that the downloader correctly follows HTTP redirects.
    """
    redirect_url = "https://example.com/final.wav"

    respx.get(TEST_URL).respond(302, headers={"Location": redirect_url})
    respx.get(redirect_url).respond(200, content=TEST_BYTES)
    

    downloaded_path = await download_audio_file(TEST_URL)

    assert downloaded_path.exists()
    assert downloaded_path.read_bytes() == TEST_BYTES
    assert downloaded_path.name.endswith(".wav") # Suffix comes from the final URL

    downloaded_path.unlink()