import asyncio
import datetime
import hashlib
import logging
import os
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any

import aiohttp

from .config import VideoConfig
from .message_parser import VideoInfo

_logger = logging.getLogger(__name__)


class VideoDownloadError(Exception):
    pass


@dataclass
class VideoResult:
    local_path: str
    file_unique: str
    file_size: int
    success: bool
    error: str = ""


def generate_video_path(
    base_dir: str, filename: str, date: datetime.datetime | None = None
) -> str:
    if date is None:
        date = datetime.datetime.now()
    date_path = os.path.join(
        base_dir, str(date.year), f"{date.month:02d}", f"{date.day:02d}"
    )
    os.makedirs(date_path, exist_ok=True)
    return os.path.abspath(os.path.join(date_path, filename))


_CONTENT_TYPE_MAP: dict[str, str] = {
    "video/mp4": "mp4",
    "video/quicktime": "mov",
    "video/x-msvideo": "avi",
    "video/x-matroska": "mkv",
    "video/webm": "webm",
}


def detect_video_extension(url: str = "", content_type: str = "") -> str:
    if url:
        match = re.search(r"\.(mp4|mov|avi|mkv|webm)", url, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    if content_type:
        mime = content_type.split(";")[0].strip().lower()
        return _CONTENT_TYPE_MAP.get(mime, "mp4")
    return "mp4"


def generate_video_filename(
    file_unique: str,
    url: str = "",
    content_type: str = "",
) -> str:
    base = (
        file_unique
        if file_unique and file_unique != "0"
        else f"{int(datetime.datetime.now().timestamp()):x}_{os.urandom(4).hex()}"
    )
    ext = detect_video_extension(url, content_type)
    sanitized_base = re.sub(r"[^\w\-_.]", "_", base)
    return f"{sanitized_base}.{ext}"


async def download_video(
    url: str,
    *,
    timeout: int = 60,
    max_size: int = 524288000,
    session: aiohttp.ClientSession | None = None,
) -> tuple[bytes, dict]:
    own_session = session is None

    # NapCat may provide a file:// URI or a local Windows path instead of HTTP.
    if not url.startswith(("http://", "https://")):
        local_path = decode_local_path(url)
        if local_path and os.path.isfile(local_path):
            data = await asyncio.to_thread(_read_file_bytes, local_path)
            return data, {"Content-Type": detect_video_mime(local_path)}
        raise VideoDownloadError(f"Local file not found: {local_path or url}")

    try:
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        if own_session:
            session = aiohttp.ClientSession(timeout=timeout_obj)
        assert session is not None
        async with session.get(url) as response:
            response.raise_for_status()
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_size:
                raise VideoDownloadError(
                    f"File too large: {content_length} bytes > {max_size} bytes"
                )
            chunks: list[bytes] = []
            total_bytes = 0
            async for chunk in response.content.iter_chunked(128 * 1024):
                total_bytes += len(chunk)
                if total_bytes > max_size:
                    raise VideoDownloadError(
                        f"File too large: {total_bytes} bytes > {max_size} bytes"
                    )
                chunks.append(chunk)
            return b"".join(chunks), dict(response.headers)
    except Exception as exc:
        raise VideoDownloadError(f"Download failed: {str(exc)}") from exc
    finally:
        if own_session and session is not None:
            await session.close()


def detect_video_mime(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    mime_map = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
    }
    return mime_map.get(ext, "video/mp4")


def decode_local_path(value: str) -> str:
    """Resolve a `file://` URI or raw filesystem string to a usable local path.

    NapCat may return any of:
      * `file:///C:/path/to/video.mp4`
      * `C:\\Users\\...\\video.mp4` (raw Windows path)
      * `/var/lib/.../video.mp4` (POSIX)

    Returns the empty string when the input is not a local-looking path.
    """
    if not value:
        return ""
    if value.startswith("file://"):
        parsed = urllib.parse.urlparse(value)
        decoded = urllib.parse.unquote(parsed.path)
        # On Windows, urlparse produces /C:/... -- strip leading slash
        if (
            os.name == "nt"
            and len(decoded) > 2
            and decoded[0] == "/"
            and decoded[2] == ":"
        ):
            decoded = decoded[1:]
        return decoded
    if value.startswith(("http://", "https://")):
        return ""
    return urllib.parse.unquote(value)


def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as file_obj:
        return file_obj.read()


def looks_like_napcat_file_code(value: str) -> bool:
    """A NapCat OB11 file code is the raw fileName: no path separators, no scheme."""
    if not value:
        return False
    if value.startswith(("http://", "https://", "file://")):
        return False
    return not any(sep in value for sep in ("/", "\\", ":"))


# Backwards-compatible alias (older callers / tests may import the underscored
# name).  Kept as a thin reference, not a re-import, to avoid mid-function imports.
_looks_like_napcat_file_code = looks_like_napcat_file_code


async def resolve_via_napcat(
    api: Any,
    file_code: str,
) -> tuple[str, str]:
    """Ask NapCat to materialize a video referenced by its OB11 file code.

    NapCat (per `napcat.mjs` GetFile action) decodes the file code (which is the
    raw fileName from the `videoElement` OB11 converter) via its in-memory UUID
    cache, then forces `FileApi.downloadMedia(...)` to materialize the file on
    disk and returns the real `file` path and remote `url`.

    Returns (local_path, url). Either may be empty if NapCat could not resolve it.
    """
    if api is None or not file_code:
        return "", ""
    get_file = _get_file_callable(api)
    if get_file is None:
        _logger.warning(
            "napcat get_file: callable not found on api file_code=%s", file_code
        )
        return "", ""
    try:
        result = await get_file(file_code)
    except Exception as exc:
        _logger.warning(
            "napcat get_file: call failed file_code=%s error=%s", file_code, exc
        )
        return "", ""
    local_path = str(getattr(result, "file", "") or "")
    remote_url = str(getattr(result, "url", "") or "")
    file_size = getattr(result, "file_size", None)
    # NapCat mirrors `file` into `url` when only a local path is available.
    if remote_url and remote_url == local_path:
        remote_url = ""
    _logger.info(
        "napcat get_file: resolved file_code=%s file=%r url=%r file_size=%r",
        file_code,
        local_path,
        remote_url,
        file_size,
    )
    return local_path, remote_url


def _get_file_callable(api: Any):
    """Locate the ``get_file`` callable on a BotAPIClient-like object.

    Real NcatBot exposes it at ``api.qq.query.get_file``; tests may attach a
    flattened ``api.qq.get_file`` for convenience.
    """
    qq = getattr(api, "qq", None)
    if qq is None:
        return None
    query = getattr(qq, "query", None)
    if query is not None and callable(getattr(query, "get_file", None)):
        return query.get_file
    if callable(getattr(qq, "get_file", None)):
        return qq.get_file
    return None


async def save_video(video_data: bytes, filepath: str) -> str:
    def _save():
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as file_obj:
            file_obj.write(video_data)
        return filepath

    return await asyncio.to_thread(_save)


async def process_video(
    video_info: VideoInfo,
    config_storage_dir: str,
    config_video: VideoConfig,
    session: aiohttp.ClientSession | None = None,
    download_semaphore: asyncio.Semaphore | None = None,
    api: Any | None = None,
) -> VideoResult:
    resolved = await _resolve_video_source(video_info, api=api)
    if isinstance(resolved, VideoResult):
        return resolved
    fetch_url = resolved

    async def _download():
        return await download_video(
            fetch_url,
            timeout=config_video.timeout,
            max_size=config_video.max_file_size,
            session=session,
        )

    try:
        if download_semaphore is None:
            video_bytes, headers = await _download()
        else:
            async with download_semaphore:
                video_bytes, headers = await _download()
        file_unique = hashlib.md5(video_bytes).hexdigest().lower()
        filename = generate_video_filename(
            file_unique,
            video_info.file_url,
            headers.get("Content-Type", ""),
        )
        filepath = generate_video_path(config_storage_dir, filename)
        await save_video(video_bytes, filepath)
        return VideoResult(
            local_path=filepath,
            file_unique=file_unique,
            file_size=len(video_bytes),
            success=True,
        )
    except VideoDownloadError as exc:
        return VideoResult(
            local_path="",
            file_unique=video_info.file_unique,
            file_size=video_info.file_size,
            success=False,
            error=str(exc),
        )


async def _resolve_video_source(
    video_info: VideoInfo,
    *,
    api: Any | None,
) -> VideoResult | str:
    """Decide how to obtain the video bytes.

    Returns either a finished :class:`VideoResult` (when the file is already
    on disk, NapCat materialized it, or fallback failed irrecoverably) or a
    plain HTTP(S) URL string to be downloaded by the caller.
    """
    # NapCat segments may include:
    #   local_path = raw fileName ("xxx.mp4") -- doubles as OB11 file code
    #   file_url   = HTTP URL, a local path, a `file://` URI, or even just a path
    #                NapCat has not yet materialized on disk
    # Try paths NapCat thinks already exist before we ask it to fetch.
    for candidate in (video_info.local_path, video_info.file_url):
        local_path = decode_local_path(str(candidate or ""))
        if local_path and os.path.isfile(local_path):
            return VideoResult(
                local_path=os.path.abspath(local_path),
                file_unique=video_info.file_unique or _hash_file(local_path),
                file_size=video_info.file_size or os.path.getsize(local_path),
                success=True,
            )

    # If we have a NapCat file code, ask NapCat to materialize the file. This is
    # the only reliable path when `getVideoUrl` fails: NapCat ships the OB11 path
    # of where it *would* put the file, but the user's QQ client has not yet
    # downloaded it.
    fetch_url = str(video_info.file_url or "")
    file_code = str(video_info.local_path or "")
    napcat_attempted = False
    if api is not None and looks_like_napcat_file_code(file_code):
        napcat_attempted = True
        resolved_path, resolved_url = await resolve_via_napcat(api, file_code)
        if resolved_path and os.path.isfile(resolved_path):
            return VideoResult(
                local_path=os.path.abspath(resolved_path),
                file_unique=video_info.file_unique or _hash_file(resolved_path),
                file_size=video_info.file_size or os.path.getsize(resolved_path),
                success=True,
            )
        # Only adopt the resolved URL when it is a usable HTTP(S) link.  NapCat
        # often mirrors the placeholder local path into `url`, which would just
        # restart the same fetch failure below.
        if resolved_url.startswith(("http://", "https://")):
            fetch_url = resolved_url
        else:
            return VideoResult(
                local_path="",
                file_unique=video_info.file_unique,
                file_size=video_info.file_size,
                success=False,
                error=(
                    "NapCat get_file returned no downloadable target "
                    f"(file={resolved_path!r}, url={resolved_url!r})"
                ),
            )

    # Without a NapCat fallback, refuse to treat a non-HTTP path as a URL.  The
    # legacy `download_video` path tolerates `file://` and absolute paths, but
    # those only succeed when the QQ client has already downloaded the video --
    # which is exactly the case the recorder must handle through NapCat.
    if not fetch_url:
        return VideoResult(
            local_path="",
            file_unique=video_info.file_unique,
            file_size=video_info.file_size,
            success=False,
            error="Empty URL",
        )
    if not fetch_url.startswith(("http://", "https://")):
        reason = (
            "NapCat fallback unavailable"
            if not napcat_attempted
            else "NapCat fallback exhausted"
        )
        return VideoResult(
            local_path="",
            file_unique=video_info.file_unique,
            file_size=video_info.file_size,
            success=False,
            error=f"No HTTP URL for video ({reason}); raw={fetch_url!r}",
        )
    return fetch_url


async def process_videos(
    videos: list[VideoInfo],
    config_storage_dir: str,
    config_video: VideoConfig,
    download_semaphore: asyncio.Semaphore | None = None,
    api: Any | None = None,
) -> list[VideoResult]:
    results: list[VideoResult] = []
    if not config_video.download:
        for video in videos:
            results.append(
                VideoResult(
                    local_path=video.local_path,
                    file_unique=video.file_unique,
                    file_size=video.file_size,
                    success=False,
                    error="Download disabled",
                )
            )
        return results

    timeout_obj = aiohttp.ClientTimeout(total=config_video.timeout)
    async with aiohttp.ClientSession(timeout=timeout_obj) as session:
        for video in videos:
            results.append(
                await process_video(
                    video,
                    config_storage_dir,
                    config_video,
                    session=session,
                    download_semaphore=download_semaphore,
                    api=api,
                )
            )
    return results


def _hash_file(path: str) -> str:
    with open(path, "rb") as file_obj:
        return hashlib.md5(file_obj.read()).hexdigest().lower()
