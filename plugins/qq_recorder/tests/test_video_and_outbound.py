import asyncio
from pathlib import Path
from types import SimpleNamespace

from plugins.grok.infra.recorder_bridge import RecorderBridge as GrokRecorderBridge
from plugins.qq_recorder.config import build_config
from plugins.qq_recorder.processors import MessageProcessor
from plugins.qq_recorder.storage import MessageStorage


class _DummyLogger:
    def info(self, *_args, **_kwargs) -> None:
        return None

    def warning(self, *_args, **_kwargs) -> None:
        return None

    def error(self, *_args, **_kwargs) -> None:
        return None


class _DummyAPI:
    class qq:
        class query:
            @staticmethod
            async def get_forward_msg(_forward_id: str) -> dict:
                return {"messages": []}


def test_message_processor_persists_and_downloads_small_video(tmp_path, monkeypatch):
    async def _run():
        db_path = tmp_path / "recorder.db"
        videos_dir = tmp_path / "videos"
        settings = build_config(
            {
                "monitor_all": True,
                "storage": {
                    "database": str(db_path),
                    "videos_dir": str(videos_dir),
                },
                "image": {"download": False},
                "video": {
                    "download": True,
                    "timeout": 5,
                    "max_file_size": 10_000_000,
                    "max_duration_sec": 120,
                },
                "backup": {"enabled": False},
            }
        )
        storage = MessageStorage(str(db_path))
        await storage.init_db()
        processor = MessageProcessor(storage, settings, _DummyAPI(), _DummyLogger())

        async def _fake_download(*_args, **_kwargs):
            return b"video-bytes", {"Content-Type": "video/mp4"}

        monkeypatch.setattr(
            "plugins.qq_recorder.video_handler.download_video",
            _fake_download,
        )

        message_id = await processor.process_message(
            {
                "message_type": "group",
                "message_id": "m-video-1",
                "user_id": "u1",
                "group_id": "g1",
                "time": 1_712_345_678,
                "raw_message": "[CQ:video,url=https://example.test/demo.mp4]",
                "message": [
                    {
                        "type": "video",
                        "data": {
                            "url": "https://example.test/demo.mp4",
                            "file_size": 2048,
                            "duration": 31,
                            "title": "demo",
                            "desc": "intro",
                        },
                    }
                ],
                "sender": {"nickname": "tester", "card": ""},
            }
        )
        stored = await storage.get_message("m-video-1")
        await storage.close()
        return message_id, stored

    message_id, stored = asyncio.run(_run())

    assert message_id is not None
    assert stored is not None
    assert len(stored.videos) == 1
    assert stored.videos[0].downloaded is True
    assert stored.videos[0].duration_sec == 31
    assert stored.videos[0].title == "demo"
    assert stored.videos[0].intro == "intro"
    assert stored.videos[0].local_path
    assert Path(stored.videos[0].local_path).exists()
    assert any(segment.segment_type == "video" for segment in stored.segments)


def test_message_processor_skips_video_download_when_over_threshold(
    tmp_path, monkeypatch
):
    async def _run():
        db_path = tmp_path / "recorder.db"
        settings = build_config(
            {
                "monitor_all": True,
                "storage": {
                    "database": str(db_path),
                    "videos_dir": str(tmp_path / "videos"),
                },
                "image": {"download": False},
                "video": {
                    "download": True,
                    "timeout": 5,
                    "max_file_size": 1024,
                    "max_duration_sec": 10,
                },
                "backup": {"enabled": False},
            }
        )
        storage = MessageStorage(str(db_path))
        await storage.init_db()
        processor = MessageProcessor(storage, settings, _DummyAPI(), _DummyLogger())

        async def _fail_download(*_args, **_kwargs):
            raise AssertionError("video download should be skipped")

        monkeypatch.setattr(
            "plugins.qq_recorder.video_handler.download_video",
            _fail_download,
        )

        message_id = await processor.process_message(
            {
                "message_type": "group",
                "message_id": "m-video-2",
                "user_id": "u1",
                "group_id": "g1",
                "time": 1_712_345_678,
                "raw_message": "[CQ:video,url=https://example.test/huge.mp4]",
                "message": [
                    {
                        "type": "video",
                        "data": {
                            "url": "https://example.test/huge.mp4",
                            "file_size": 4096,
                            "duration": 99,
                            "title": "huge",
                        },
                    }
                ],
                "sender": {"nickname": "tester", "card": ""},
            }
        )
        stored = await storage.get_message("m-video-2")
        await storage.close()
        return message_id, stored

    message_id, stored = asyncio.run(_run())

    assert message_id is not None
    assert stored is not None
    assert len(stored.videos) == 1
    assert stored.videos[0].downloaded is False
    assert stored.videos[0].local_path is None


def test_outbound_group_send_is_recorded_and_reply_chain_can_reach_it(tmp_path):
    from plugins.qq_recorder.outbound_recorder import install_outbound_recording

    class _RawQQAPI:
        async def send_group_msg(self, group_id, message, **kwargs):
            del kwargs
            assert group_id == "30001"
            assert message[0]["type"] == "reply"
            return SimpleNamespace(message_id="bot-msg-1")

        async def send_private_msg(self, user_id, message, **kwargs):
            del user_id, message, kwargs
            raise AssertionError("private send not expected")

        async def send_forward_msg(self, message_type, target_id, messages, **kwargs):
            del message_type, target_id, messages, kwargs
            raise AssertionError("forward send not expected")

    async def _run():
        db_path = tmp_path / "recorder.db"
        storage = MessageStorage(str(db_path))
        await storage.init_db()

        api = SimpleNamespace(qq=SimpleNamespace(_api=_RawQQAPI()))
        install_outbound_recording(
            api,
            storage,
            bot_uin="10000",
            logger=_DummyLogger(),
        )

        await api.qq._api.send_group_msg(
            "30001",
            [
                {"type": "reply", "data": {"id": "src-1"}},
                {"type": "text", "data": {"text": "bot says hello"}},
            ],
        )

        settings = build_config(
            {
                "monitor_all": True,
                "storage": {"database": str(db_path)},
                "image": {"download": False},
                "backup": {"enabled": False},
            }
        )
        processor = MessageProcessor(storage, settings, _DummyAPI(), _DummyLogger())
        await processor.process_message(
            {
                "message_type": "group",
                "message_id": "user-reply-1",
                "user_id": "20001",
                "group_id": "30001",
                "time": 1_712_345_679,
                "raw_message": "[CQ:reply,id=bot-msg-1] what do you mean",
                "message": [
                    {"type": "reply", "data": {"id": "bot-msg-1"}},
                    {"type": "text", "data": {"text": "what do you mean"}},
                ],
                "sender": {"nickname": "user", "card": ""},
            }
        )

        outbound = await storage.get_message("bot-msg-1")
        bridge = GrokRecorderBridge()
        await bridge.connect_existing(str(db_path))
        source_msg = await bridge.get_message("user-reply-1")
        assert source_msg is not None
        chain = await bridge.get_reply_chain(source_msg, max_depth=2)
        await bridge.close()
        await storage.close()
        return outbound, chain

    outbound, chain = asyncio.run(_run())

    assert outbound is not None
    assert outbound.user_id == "10000"
    assert outbound.group_id == "30001"
    assert outbound.replies[0].reply_to_message_id == "src-1"
    assert "bot says hello" in outbound.raw_message
    assert len(chain) == 2
    assert chain[0].message_id == "user-reply-1"
    assert "bot says hello" in chain[1].raw_message


def test_message_processor_uses_napcat_get_file_when_local_path_missing(
    tmp_path, monkeypatch
):
    """When NapCat's segment carries a bare file code but no live file, the
    recorder should ask NapCat (via `get_file`) to materialize the video and
    then ingest the resolved local path."""

    async def _run():
        db_path = tmp_path / "recorder.db"
        videos_dir = tmp_path / "videos"
        videos_dir.mkdir()

        # Simulate the file NapCat creates only after we ask for it.
        resolved_file = tmp_path / "napcat_resolved.mp4"
        resolved_file.write_bytes(b"video-bytes-from-napcat")

        settings = build_config(
            {
                "monitor_all": True,
                "storage": {
                    "database": str(db_path),
                    "videos_dir": str(videos_dir),
                },
                "image": {"download": False},
                "video": {
                    "download": True,
                    "timeout": 5,
                    "max_file_size": 10_000_000,
                    "max_duration_sec": 120,
                },
                "backup": {"enabled": False},
            }
        )
        storage = MessageStorage(str(db_path))
        await storage.init_db()

        get_file_calls: list[str] = []

        class _GetFileResult:
            def __init__(self, file: str, url: str = "") -> None:
                self.file = file
                self.url = url

        class _NapcatAPI:
            class qq:
                class query:
                    @staticmethod
                    async def get_forward_msg(_forward_id: str) -> dict:
                        return {"messages": []}

                @staticmethod
                async def get_file(file_id: str):
                    get_file_calls.append(file_id)
                    return _GetFileResult(file=str(resolved_file))

        async def _fail_download(*_args, **_kwargs):
            raise AssertionError(
                "HTTP download should not run when NapCat resolves the file"
            )

        monkeypatch.setattr(
            "plugins.qq_recorder.video_handler.download_video",
            _fail_download,
        )

        processor = MessageProcessor(storage, settings, _NapcatAPI(), _DummyLogger())

        message_id = await processor.process_message(
            {
                "message_type": "group",
                "message_id": "m-video-napcat",
                "user_id": "u1",
                "group_id": "g1",
                "time": 1_712_345_678,
                "raw_message": (
                    "[CQ:video,file=deadbeef0123456789.mp4,"
                    "url=C:/Users/test/Tencent/Video/Ori/deadbeef0123456789.mp4]"
                ),
                "message": [
                    {
                        "type": "video",
                        "data": {
                            "file": "deadbeef0123456789.mp4",
                            # NapCat's placeholder local path; the QQ client has
                            # not actually downloaded the file there yet.
                            "url": (
                                "C:/Users/test/Tencent/Video/Ori/deadbeef0123456789.mp4"
                            ),
                            "file_size": len(b"video-bytes-from-napcat"),
                        },
                    }
                ],
                "sender": {"nickname": "tester", "card": ""},
            }
        )
        stored = await storage.get_message("m-video-napcat")
        await storage.close()
        return message_id, stored, get_file_calls

    message_id, stored, get_file_calls = asyncio.run(_run())

    assert message_id is not None
    assert get_file_calls == ["deadbeef0123456789.mp4"]
    assert stored is not None
    assert len(stored.videos) == 1
    assert stored.videos[0].downloaded is True
    assert stored.videos[0].local_path
    assert Path(stored.videos[0].local_path).exists()
