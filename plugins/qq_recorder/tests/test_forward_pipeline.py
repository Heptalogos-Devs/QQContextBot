import asyncio
from datetime import datetime

from plugins.qq_recorder.config import build_config
from plugins.qq_recorder.forward_parser import parse_forward_response
from plugins.qq_recorder.processors import MessageProcessor
from plugins.qq_recorder.storage import MessageStorage


class _ForwardAPI:
    class qq:
        class query:
            @staticmethod
            async def get_forward_msg(_forward_id: str) -> dict:
                return {
                    "messages": [
                        {
                            "sender": {"user_id": "20001", "nickname": "阿梓"},
                            "content": [
                                {"type": "text", "data": {"text": "第一段"}},
                                {
                                    "type": "image",
                                    "data": {"url": "https://example/a.png"},
                                },
                            ],
                        }
                    ]
                }


class _DummyLogger:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def info(self, msg, *args, **kwargs) -> None:
        self.infos.append(str(msg) % args if args else str(msg))

    def warning(self, msg, *args, **kwargs) -> None:
        self.warnings.append(str(msg) % args if args else str(msg))

    def error(self, *_args, **_kwargs) -> None:
        return None


def test_parse_forward_response_supports_sender_content_shape():
    nodes = parse_forward_response(
        {
            "messages": [
                {
                    "sender": {"user_id": "20001", "nickname": "阿梓"},
                    "content": [
                        {"type": "text", "data": {"text": "第一段"}},
                        {"type": "image", "data": {"url": "https://example/a.png"}},
                    ],
                }
            ]
        }
    )

    assert len(nodes) == 1
    assert nodes[0].user_id == "20001"
    assert nodes[0].nickname == "阿梓"
    assert nodes[0].content_summary == "第一段[图片]"


def test_message_processor_persists_forward_messages_from_sender_content_shape(
    tmp_path,
):
    async def _run():
        db_path = tmp_path / "recorder.db"
        settings = build_config(
            {
                "monitor_all": True,
                "image": {"download": False},
                "forward": {"parse_content": True},
                "backup": {"enabled": False},
                "storage": {"database": str(db_path)},
            }
        )
        storage = MessageStorage(str(db_path))
        await storage.init_db()
        logger = _DummyLogger()
        processor = MessageProcessor(storage, settings, _ForwardAPI(), logger)

        message_id = await processor.process_message(
            {
                "message_type": "group",
                "message_id": "m-forward-1",
                "user_id": "u1",
                "group_id": "g1",
                "time": int(datetime.now().timestamp()),
                "raw_message": "[CQ:forward,id=fwd-1]",
                "message": [{"type": "forward", "data": {"id": "fwd-1"}}],
                "sender": {"nickname": "tester", "card": ""},
            }
        )
        # Forward API backfill runs via ensure_future — yield to the event
        # loop so it completes before we read the stored message.
        await asyncio.sleep(0)
        stored = await storage.get_message("m-forward-1")
        await storage.close()
        return message_id, stored, logger

    message_id, stored, logger = asyncio.run(_run())

    assert message_id is not None
    assert stored is not None
    assert stored.has_forward is True
    assert len(stored.forward_messages) == 1
    assert stored.forward_messages[0].user_id == "20001"
    assert stored.forward_messages[0].nickname == "阿梓"
    assert stored.forward_messages[0].content_summary == "第一段[图片]"
    assert any("flatten_count=1" in line for line in logger.infos)
