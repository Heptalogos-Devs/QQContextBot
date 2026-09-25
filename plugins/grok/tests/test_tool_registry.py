import asyncio
import json
from pathlib import Path

import pytest

from plugins.grok.app.runtime import AgentRuntime
from plugins.grok.tools.guide_tools import build_load_tool_guide_tool
from plugins.grok.tools.registry import (
    ToolArgumentError,
    ToolDefinition,
    ToolRegistry,
)


async def _echo_tool(_context, arguments):
    return {"status": "ok", "data": {"echo": arguments["query"]}}


def _tool_schema_payloads():
    tools_dir = Path(__file__).resolve().parent.parent / "schemas" / "tools"
    for path in sorted(tools_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        yield path.stem, json.loads(path.read_text(encoding="utf-8"))


def test_tool_registry_rejects_unknown_arguments():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="load_context",
            description="Load recent context for the current chat scope.",
            schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
            handler=_echo_tool,
        )
    )

    with pytest.raises(ToolArgumentError, match="unknown argument"):
        registry.validate_tool_call("load_context", {"query": "topic", "extra": True})


def test_tool_registry_rejects_missing_required_argument():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="load_context",
            description="Load recent context for the current chat scope.",
            schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
            handler=_echo_tool,
        )
    )

    with pytest.raises(ToolArgumentError, match="missing required argument"):
        registry.validate_tool_call("load_context", {"limit": 3})


def test_tool_registry_executes_registered_tool():
    async def _run():
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="load_context",
                description="Load recent context for the current chat scope.",
                schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["query"],
                },
                handler=_echo_tool,
            )
        )

        result = await registry.execute(
            "load_context",
            {"query": "topic", "limit": 5},
            context={"chat_type": "group", "chat_id": "30001"},
        )

        assert result["status"] == "ok"
        assert result["data"]["echo"] == "topic"

    asyncio.run(_run())


def test_agent_runtime_registry_exposes_terminate_and_load_tool_guide_tools():
    plugin = type(
        "PluginStub",
        (),
        {
            "settings": type(
                "SettingsStub",
                (),
                {
                    "agent": type(
                        "AgentStub",
                        (),
                        {
                            "max_steps": 3,
                            "max_tool_calls_per_turn": 2,
                        },
                    )(),
                    "vision": type(
                        "VisionStub",
                        (),
                        {
                            "api_image_bytes_max": 1024,
                            "image_fast_model": "image-model",
                            "video_summary_model": "video-model",
                            "prompt_version": "prompt-v1",
                            "schema_version": "schema-v1",
                        },
                    )(),
                },
            )(),
            "_bridge": None,
            "_vision_client": None,
            "_vision_quota": None,
            "_profile_json_store": None,
        },
    )()

    runtime = AgentRuntime(plugin)
    tool_names = {
        item["function"]["name"] for item in runtime.registry.list_for_model()
    }

    assert "terminate" in tool_names
    assert "load_tool_guide" in tool_names
    assert "query_chat_history" in tool_names


def test_tool_registry_strips_internal_prompt_metadata_from_model_schema():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="read_picture",
            description="Analyze an image attached to a recorded message.",
            schema={
                "type": "object",
                "description": "Top-level schema purpose.",
                "additionalProperties": False,
                "properties": {
                    "message_id": {
                        "type": "string",
                        "description": "Message ID to inspect.",
                    }
                },
                "x-prompt": {
                    "summary": "internal summary",
                    "usage": ["internal usage"],
                },
            },
            handler=_echo_tool,
        )
    )

    exposed = registry.list_for_model()[0]["function"]["parameters"]

    assert "x-prompt" not in exposed
    assert "x-tool-meta" not in exposed
    assert (
        exposed["properties"]["message_id"]["description"] == "Message ID to inspect."
    )


def test_load_tool_guide_tool_returns_full_guidance_for_one_tool():
    async def _run():
        tool = build_load_tool_guide_tool(None)

        result = await tool.handler({}, {"tool_name": "terminate"})

        assert result.status == "ok"
        assert result.data["tool_name"] == "terminate"
        assert "结束本次对话" in result.data["summary"]
        assert any("不会发送任何消息" in item for item in result.data["boundaries"])
        assert result.data["policy"]["full_exposure"] is True
        assert result.data["policy"]["counts_against_budget"] is False
        assert result.data["policy"]["same_arguments_limit"] == "per_agent_run"
        assert any("不消耗工具调用额度" in item for item in result.data["policy_hints"])

    asyncio.run(_run())


def test_load_tool_guide_tool_rejects_unknown_tool_name():
    async def _run():
        tool = build_load_tool_guide_tool(None)

        result = await tool.handler({}, {"tool_name": "missing_tool"})

        assert result.status == "failed"
        assert result.error_code == "unknown_tool_guide"

    asyncio.run(_run())


def test_all_tool_schemas_expose_complete_prompt_guidance_without_impl_leaks():
    banned_phrases = [
        "JSON 文件",
        "动态注入",
        "视觉 AI 模型",
        "内部调试",
        "内部 trace",
        "系统会自动",
        "下一个回合",
    ]

    for tool_name, payload in _tool_schema_payloads():
        meta = payload.get("x-prompt", {}) or {}
        tool_meta = payload.get("x-tool-meta", {}) or {}

        assert meta.get("summary"), f"{tool_name} missing x-prompt.summary"
        assert meta.get("usage"), f"{tool_name} missing x-prompt.usage"
        assert meta.get("guidance"), f"{tool_name} missing x-prompt.guidance"
        assert meta.get("boundaries"), f"{tool_name} missing x-prompt.boundaries"
        assert "full_exposure" in tool_meta, (
            f"{tool_name} missing x-tool-meta.full_exposure"
        )
        assert "counts_against_budget" in tool_meta, (
            f"{tool_name} missing x-tool-meta.counts_against_budget"
        )
        assert tool_meta.get("same_arguments_limit") in {"none", "per_agent_run"}, (
            f"{tool_name} has invalid x-tool-meta.same_arguments_limit"
        )

        text_parts = [str(payload.get("description", "") or "")]
        for prop in (payload.get("properties", {}) or {}).values():
            if isinstance(prop, dict):
                text_parts.append(str(prop.get("description", "") or ""))
        for key in ("summary",):
            text_parts.append(str(meta.get(key, "") or ""))
        for key in ("usage", "guidance", "boundaries"):
            text_parts.extend(str(item or "") for item in (meta.get(key, []) or []))

        joined = "\n".join(text_parts)
        for phrase in banned_phrases:
            assert phrase not in joined, (
                f"{tool_name} leaks implementation detail: {phrase}"
            )
