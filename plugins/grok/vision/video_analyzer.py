from __future__ import annotations

import asyncio
import base64
import json
import logging
import urllib.parse
from datetime import datetime
from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError

from .video_prompts import build_video_messages
from .video_schemas import VideoAnalysis, normalize_video_analysis

logger = logging.getLogger("grok.vision.video_analyzer")


async def analyze_video(
    client,
    video_file_path: str | None,
    video_url: str | None,
    file_unique: str,
    settings,
    *,
    chat_context: str = "",
    title: str = "",
    intro: str = "",
) -> VideoAnalysis:
    model = settings.vision.video_summary_model
    temperature = settings.vision.temperature
    timeout = settings.vision.video_timeout_sec

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    messages = build_video_messages(
        chat_context=chat_context,
        current_time=current_time,
        title=title,
        intro=intro,
    )
    attach_error = await _attach_video_content(messages, video_file_path, video_url)
    if attach_error is not None:
        return attach_error

    start = datetime.now()
    try:
        completion = await _call_video_model(
            client, model, messages, temperature, timeout
        )
    except APITimeoutError as exc:
        elapsed = (datetime.now() - start).total_seconds()
        logger.warning(
            "video: timeout file_unique=%s model=%s elapsed=%.1fs error=%s",
            file_unique,
            model,
            elapsed,
            exc,
        )
        return VideoAnalysis(error_code="video_timeout")
    except APIConnectionError as exc:
        elapsed = (datetime.now() - start).total_seconds()
        logger.warning(
            "video: connection error file_unique=%s model=%s elapsed=%.1fs error=%s",
            file_unique,
            model,
            elapsed,
            exc,
        )
        return VideoAnalysis(error_code="video_connection_error")
    except APIStatusError as exc:
        elapsed = (datetime.now() - start).total_seconds()
        logger.warning(
            "video: api status error file_unique=%s model=%s status=%s elapsed=%.1fs",
            file_unique,
            model,
            exc.status_code,
            elapsed,
        )
        return VideoAnalysis(error_code="video_api_error")
    except Exception as exc:
        elapsed = (datetime.now() - start).total_seconds()
        logger.warning(
            "video: model call failed file_unique=%s model=%s elapsed=%.1fs error=%s",
            file_unique,
            model,
            elapsed,
            exc,
        )
        return VideoAnalysis(error_code="video_llm_error")

    raw_content = _extract_content(completion)
    if not raw_content:
        return VideoAnalysis(error_code="video_empty_response")

    try:
        data = json.loads(raw_content)
    except (TypeError, json.JSONDecodeError) as exc:
        logger.warning(
            "video: invalid JSON from model file_unique=%s error=%s",
            file_unique,
            exc,
        )
        return VideoAnalysis(
            error_code="video_invalid_json",
            raw_model_output=raw_content,
        )

    if not isinstance(data, dict):
        return VideoAnalysis(
            error_code="video_invalid_json",
            raw_model_output=raw_content,
        )

    analysis = normalize_video_analysis(data, raw_model_output=raw_content)
    if not analysis.error_code:
        elapsed = (datetime.now() - start).total_seconds()
        logger.info(
            "video: success file_unique=%s model=%s confidence=%.2f elapsed=%.1fs",
            file_unique,
            model,
            analysis.confidence,
            elapsed,
        )
    return analysis


def _extract_content(completion) -> str:
    try:
        choices = completion.choices
        if choices:
            content = choices[0].message.content
            if content is not None:
                return str(content)
    except AttributeError:
        pass
    return ""


async def _attach_video_content(
    messages: list[dict[str, str | list]],
    video_file_path: str | None,
    video_url: str | None,
) -> VideoAnalysis | None:
    user_msg = messages[1]
    assert isinstance(user_msg["content"], list)

    # file_url from NapCat may be a local path (URL-encoded Windows path).
    # Try it as both the explicit file_path and as a fallback for video_url.
    local_candidates: list[str] = []
    if video_file_path:
        local_candidates.append(video_file_path)
    if video_url and not video_url.startswith(("http://", "https://")):
        local_candidates.append(urllib.parse.unquote(video_url))

    for path in local_candidates:
        try:
            data = await asyncio.to_thread(Path(path).read_bytes)
            video_b64 = base64.b64encode(data).decode("ascii")
            user_msg["content"].append(
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
                }
            )
            return None
        except Exception:
            continue

    # Only forward genuine HTTP(S) URLs to the model; otherwise the provider
    # will reject the request (e.g. status=400) and the failure looks like a
    # vision error instead of a missing-bytes problem.
    if video_url and video_url.startswith(("http://", "https://")):
        user_msg["content"].append(
            {
                "type": "video_url",
                "video_url": {"url": video_url},
            }
        )
        return None

    return VideoAnalysis(error_code="video_no_source")


async def _call_video_model(
    client,
    model: str,
    messages: list[dict[str, str | list]],
    temperature: float,
    timeout: int,
):
    def _call():
        return client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
            timeout=timeout,
        )

    return await asyncio.to_thread(_call)
