from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from ..shared import load_schema
from .context_tools import _message_payload
from .registry import ToolDefinition, ToolResponse

logger = logging.getLogger("grok.history_tools")


def build_history_tools(plugin) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="query_chat_history",
            description=(
                "Query recorded chat history by filters such as user, keyword, "
                "time, and message traits."
            ),
            schema=load_schema("tools/query_chat_history.json"),
            handler=_query_chat_history_handler(plugin),
        )
    ]


def _query_chat_history_handler(plugin):
    async def _handler(
        context: dict[str, Any],
        arguments: dict[str, Any],
    ) -> ToolResponse:
        bridge = plugin._bridge
        source_msg = context.get("source_msg")
        if bridge is None or source_msg is None:
            return ToolResponse(
                status="failed",
                data={},
                error_code="message_not_found",
                message="source message not available",
            )

        explicit_chat_id = str(arguments.get("chat_id") or "").strip()
        explicit_chat_type = str(arguments.get("chat_type") or "").strip()
        if explicit_chat_id and not explicit_chat_type:
            return ToolResponse(
                status="failed",
                data={},
                error_code="chat_type_required",
                message="chat_id 需要和 chat_type 一起传入",
                retryable=False,
            )

        chat_type = explicit_chat_type or str(context.get("chat_type") or "")
        chat_id = explicit_chat_id or str(context.get("chat_id") or "")
        query = {
            "user_id": _optional_str(arguments.get("user_id")),
            "chat_type": chat_type or None,
            "chat_id": chat_id or None,
            "keyword": _optional_str(arguments.get("keyword")),
            "time_from": _optional_str(arguments.get("time_from")),
            "time_to": _optional_str(arguments.get("time_to")),
            "has_forward": _optional_bool(arguments, "has_forward"),
            "has_image": _optional_bool(arguments, "has_image"),
            "has_reply": _optional_bool(arguments, "has_reply"),
            "has_video": _optional_bool(arguments, "has_video"),
            "has_at": _optional_bool(arguments, "has_at"),
            "has_app_share": _optional_bool(arguments, "has_app_share"),
            "limit": max(1, int(arguments.get("limit", 20) or 20)),
            "order": str(arguments.get("order", "desc") or "desc").lower(),
        }

        if not _has_effective_filters(arguments):
            return ToolResponse(
                status="failed",
                data={"query": query},
                error_code="empty_history_query",
                message=(
                    "这个查询没有任何筛选条件；如果只是想看当前附近上下文，"
                    "请改用 load_context"
                ),
                retryable=False,
            )
        time_error = _validate_time_filters(query["time_from"], query["time_to"])
        if time_error:
            return ToolResponse(
                status="failed",
                data={"query": query},
                error_code="invalid_time_filter",
                message=time_error,
                retryable=False,
            )

        logger.info(
            "query_chat_history scope=%s:%s user_id=%s keyword=%s limit=%d order=%s",
            query["chat_type"] or "",
            query["chat_id"] or "",
            query["user_id"] or "",
            query["keyword"] or "",
            query["limit"],
            query["order"],
        )
        try:
            items = await bridge.query_chat_history(**query)
        except Exception as exc:
            logger.warning(
                "query_chat_history failed scope=%s:%s error=%s",
                query["chat_type"] or "",
                query["chat_id"] or "",
                exc,
            )
            return ToolResponse(
                status="failed",
                data={"query": query},
                error_code="history_query_failed",
                message=str(exc),
            )

        payload_messages = [await _message_payload(bridge, item) for item in items]
        return ToolResponse(
            status="ok",
            data={
                "messages": payload_messages,
                "query": query,
                "total_returned": len(payload_messages),
            },
        )

    return _handler


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _optional_bool(arguments: dict[str, Any], key: str) -> bool | None:
    if key not in arguments:
        return None
    value = arguments.get(key)
    if isinstance(value, bool):
        return value
    return None if value is None else None


def _has_effective_filters(arguments: dict[str, Any]) -> bool:
    return any(
        arguments.get(key) not in ("", None)
        for key in (
            "user_id",
            "keyword",
            "time_from",
            "time_to",
            "has_forward",
            "has_image",
            "has_reply",
            "has_video",
            "has_at",
            "has_app_share",
        )
    )


def _validate_time_filters(time_from: str | None, time_to: str | None) -> str:
    start = _try_parse_time(time_from)
    if time_from and start is None:
        return "time_from 格式无效，必须是 YYYY-MM-DD HH:MM:SS 或 ISO-8601"
    end = _try_parse_time(time_to)
    if time_to and end is None:
        return "time_to 格式无效，必须是 YYYY-MM-DD HH:MM:SS 或 ISO-8601"
    if start is not None and end is not None and start > end:
        return "time_from 不能晚于 time_to"
    return ""


def _try_parse_time(value: str | None) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
