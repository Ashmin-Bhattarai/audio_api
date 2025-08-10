from pathlib import Path
import tempfile
import uuid
from urllib.parse import urlparse

import httpx
import aiofiles
from loguru import logger


async def download_audio_file(url: str):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http(s) URLs are supported.")
    
    suffix = Path(parsed.path).suffix or ""
    temp_dir = Path(tempfile.gettempdir())
    temp_path = temp_dir / f"audio_{uuid.uuid4().hex}{suffix}"

    logger.info(f"Starting download from {url} to {temp_path}")


    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                async with aiofiles.open(temp_path, "wb") as fp:
                    async for chunk in resp.aiter_bytes():
                        await fp.write(chunk)

        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            logger.error(f"Failed to download file from {url}, Error: {repr(e)}")
            raise

        logger.success(f"Sucessfully downloaded file to {temp_path}")
        return temp_path