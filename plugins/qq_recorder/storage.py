import asyncio
import sqlite3
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from sqlalchemy import asc, desc, func, select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from .config import LockRetryConfig
from .models import (
    AppShare,
    AtMention,
    Base,
    ForwardMessage,
    Image,
    ImageAnalysis,
    Message,
    MessageSegment,
    MonitoredChat,
    Reply,
    Video,
    init_engine,
)

T = TypeVar("T")


async def _add_segments(session, message_id: int, segments: list[dict]) -> None:
    for seg in segments:
        session.add(
            MessageSegment(
                message_id=message_id,
                segment_type=seg["segment_type"],
                segment_order=seg["segment_order"],
                segment_data=seg["segment_data"],
            )
        )


async def _add_images(session, message_id: int, images: list[dict]) -> None:
    for img in images:
        session.add(
            Image(
                message_id=message_id,
                file_url=img.get("file_url"),
                file_unique=img.get("file_unique"),
                file_size=img.get("file_size"),
                local_path=img.get("local_path"),
                width=img.get("width"),
                height=img.get("height"),
                downloaded=img.get("downloaded", False),
            )
        )


async def _add_videos(session, message_id: int, videos: list[dict]) -> None:
    for video in videos:
        session.add(
            Video(
                message_id=message_id,
                file_url=video.get("file_url"),
                file_unique=video.get("file_unique"),
                file_size=video.get("file_size"),
                local_path=video.get("local_path"),
                duration_sec=video.get("duration_sec"),
                downloaded=video.get("downloaded", False),
                title=video.get("title", ""),
                intro=video.get("intro", ""),
            )
        )


async def _add_replies(session, message_id: int, replies: list[dict]) -> None:
    for reply in replies:
        session.add(
            Reply(
                message_id=message_id,
                reply_to_message_id=reply["reply_to_message_id"],
            )
        )


async def _add_forward_messages(
    session,
    message_id: int,
    forwards: list[dict],
    parent_id: int | None = None,
) -> None:
    for forward_data in forwards:
        forward = ForwardMessage(
            message_id=message_id,
            parent_forward_id=parent_id,
            user_id=forward_data.get("user_id"),
            nickname=forward_data.get("nickname"),
            depth=forward_data.get("depth", 0),
            content_summary=forward_data.get("content_summary"),
            forward_id=forward_data.get("forward_id"),
        )
        session.add(forward)
        await session.flush()
        children = forward_data.get("children", [])
        if children:
            await _add_forward_messages(
                session, message_id, children, parent_id=forward.id
            )


async def _add_at_mentions(session, message_id: int, mentions: list[dict]) -> None:
    for at in mentions:
        session.add(
            AtMention(message_id=message_id, target_user_id=at["target_user_id"])
        )


async def _add_app_shares(session, message_id: int, shares: list[dict]) -> None:
    for share in shares:
        session.add(
            AppShare(
                message_id=message_id,
                app_name=share.get("app_name", ""),
                title=share.get("title", ""),
                description=share.get("description", ""),
                url=share.get("url", ""),
                prompt=share.get("prompt", ""),
                raw_data=share.get("raw_data", ""),
            )
        )


def _apply_message_query_filters(
    stmt,
    *,
    user_id: str | None = None,
    chat_type: str | None = None,
    chat_id: str | None = None,
    keyword: str | None = None,
    time_from=None,
    time_to=None,
    has_forward: bool | None = None,
    has_image: bool | None = None,
    has_reply: bool | None = None,
    has_video: bool | None = None,
    has_at: bool | None = None,
    has_app_share: bool | None = None,
):
    direct_filters = (
        (bool(user_id), Message.user_id == user_id),
        (bool(chat_type), Message.chat_type == chat_type),
        (bool(keyword), Message.raw_message.contains(keyword or "")),
        (time_from is not None, Message.timestamp >= time_from),
        (time_to is not None, Message.timestamp <= time_to),
    )
    for enabled, expression in direct_filters:
        if enabled:
            stmt = stmt.where(expression)
    if chat_id:
        if chat_type == "group":
            stmt = stmt.where(Message.group_id == chat_id)
        elif chat_type == "private":
            stmt = stmt.where(Message.user_id == chat_id)
    for column, value in (
        (Message.has_forward, has_forward),
        (Message.has_image, has_image),
        (Message.has_reply, has_reply),
        (Message.has_video, has_video),
        (Message.has_at, has_at),
        (Message.has_app_share, has_app_share),
    ):
        if value is not None:
            stmt = stmt.where(column.is_(value))
    return stmt


class MessageStorage:
    def __init__(
        self,
        db_path: str,
        lock_retry: LockRetryConfig | None = None,
        sqlite_timeout: int = 10,
        logger: Any | None = None,
    ):
        self.db_path = db_path
        self.engine = None
        self.AsyncSessionLocal: async_sessionmaker[AsyncSession] | None = None
        self.lock_retry = lock_retry or LockRetryConfig()
        self.sqlite_timeout = sqlite_timeout
        self.logger = logger

    async def init_db(self):
        self.engine, self.AsyncSessionLocal = await init_engine(
            self.db_path, sqlite_timeout=self.sqlite_timeout
        )
        async with self.engine.begin() as conn:
            await conn.run_sync(_ensure_message_sender_columns)
            await conn.run_sync(Base.metadata.create_all)

    async def close(self):
        if self.engine:
            await self.engine.dispose()

    def _session(self) -> AsyncSession:
        """Return AsyncSessionLocal with assertion that init_db() was called."""
        assert self.AsyncSessionLocal is not None, "init_db() not called"
        return self.AsyncSessionLocal()

    @staticmethod
    def _is_locked_error(exc: OperationalError) -> bool:
        error_text = str(exc).lower()
        if (
            "database is locked" in error_text
            or "database table is locked" in error_text
        ):
            return True
        if isinstance(exc.orig, sqlite3.OperationalError):
            orig_text = str(exc.orig).lower()
            return (
                "database is locked" in orig_text
                or "database table is locked" in orig_text
            )
        return False

    async def _run_with_lock_retry(
        self,
        op_name: str,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        max_retries = self.lock_retry.max_retries
        for attempt in range(max_retries + 1):
            try:
                return await operation()
            except OperationalError as exc:
                if (
                    not self.lock_retry.enabled
                    or not self._is_locked_error(exc)
                    or attempt >= max_retries
                ):
                    raise
                delay_ms = self.lock_retry.base_delay_ms * (2**attempt)
                if self.logger:
                    self.logger.warning(
                        "Retrying DB operation %s after lock conflict "
                        "(attempt %d/%d, delay_ms=%d)",
                        op_name,
                        attempt + 1,
                        max_retries,
                        delay_ms,
                    )
                await asyncio.sleep(delay_ms / 1000)
        raise RuntimeError(f"unreachable lock retry path for {op_name}")

    async def save_message(self, message_data: dict) -> int:
        async def _save_once() -> int:
            async with self._session() as session:
                try:
                    message = Message(
                        message_id=message_data["message_id"],
                        user_id=message_data["user_id"],
                        sender_nickname=message_data.get("sender_nickname"),
                        sender_card=message_data.get("sender_card"),
                        group_id=message_data["group_id"],
                        chat_type=message_data["chat_type"],
                        timestamp=message_data["timestamp"],
                        raw_message=message_data["raw_message"],
                        has_image=len(message_data.get("images", [])) > 0,
                        has_reply=len(message_data.get("replies", [])) > 0,
                        has_forward=len(message_data.get("forward_messages", [])) > 0,
                        has_video=len(message_data.get("videos", [])) > 0,
                        has_at=len(message_data.get("at_mentions", [])) > 0,
                        has_app_share=len(message_data.get("app_shares", [])) > 0,
                    )
                    session.add(message)
                    await session.flush()

                    segments = message_data.get("segments", [])
                    await _add_segments(session, message.id, segments)
                    images = message_data.get("images", [])
                    await _add_images(session, message.id, images)
                    videos = message_data.get("videos", [])
                    await _add_videos(session, message.id, videos)
                    replies = message_data.get("replies", [])
                    await _add_replies(session, message.id, replies)
                    forwards = message_data.get("forward_messages", [])
                    await _add_forward_messages(session, message.id, forwards)
                    mentions = message_data.get("at_mentions", [])
                    await _add_at_mentions(session, message.id, mentions)
                    shares = message_data.get("app_shares", [])
                    await _add_app_shares(session, message.id, shares)

                    await session.commit()
                    return message.id
                except IntegrityError:
                    await session.rollback()
                    existing = await self.get_message(message_data["message_id"])
                    if existing is not None:
                        return existing.id
                    raise
                except Exception:
                    await session.rollback()
                    raise

        return await self._run_with_lock_retry("save_message", _save_once)

    async def get_message(self, message_id: str) -> Message | None:
        async with self._session() as session:
            stmt = (
                select(Message)
                .where(Message.message_id == message_id)
                .options(
                    selectinload(Message.segments),
                    selectinload(Message.images),
                    selectinload(Message.videos),
                    selectinload(Message.replies),
                    selectinload(Message.forward_messages).options(
                        selectinload(ForwardMessage.children)
                    ),
                    selectinload(Message.at_mentions),
                    selectinload(Message.app_shares),
                )
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def get_recent_messages(
        self, chat_type: str | None = None, chat_id: str | None = None, limit: int = 10
    ) -> list[Message]:
        async with self._session() as session:
            stmt = (
                select(Message)
                .options(
                    selectinload(Message.images),
                    selectinload(Message.videos),
                    selectinload(Message.at_mentions),
                    selectinload(Message.app_shares),
                )
                .order_by(desc(Message.timestamp))
            )
            if chat_type:
                stmt = stmt.where(Message.chat_type == chat_type)
            if chat_id:
                if chat_type == "group":
                    stmt = stmt.where(Message.group_id == chat_id)
                else:
                    stmt = stmt.where(Message.user_id == chat_id)
            stmt = stmt.limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def count_messages(
        self, chat_type: str | None = None, chat_id: str | None = None
    ) -> int:
        async with self._session() as session:
            stmt = select(func.count(Message.id))
            if chat_type:
                stmt = stmt.where(Message.chat_type == chat_type)
            if chat_id:
                if chat_type == "group":
                    stmt = stmt.where(Message.group_id == chat_id)
                else:
                    stmt = stmt.where(Message.user_id == chat_id)
            result = await session.execute(stmt)
            return result.scalar_one()

    async def count_images(
        self, chat_type: str | None = None, chat_id: str | None = None
    ) -> int:
        async with self._session() as session:
            stmt = select(func.count(Image.id)).join(
                Message, Image.message_id == Message.id
            )
            if chat_type:
                stmt = stmt.where(Message.chat_type == chat_type)
            if chat_id:
                if chat_type == "group":
                    stmt = stmt.where(Message.group_id == chat_id)
                else:
                    stmt = stmt.where(Message.user_id == chat_id)
            result = await session.execute(stmt)
            return result.scalar_one()

    async def count_downloaded_images(
        self, chat_type: str | None = None, chat_id: str | None = None
    ) -> int:
        async with self._session() as session:
            stmt = (
                select(func.count(Image.id))
                .join(Message, Image.message_id == Message.id)
                .where(Image.downloaded)
            )
            if chat_type:
                stmt = stmt.where(Message.chat_type == chat_type)
            if chat_id:
                if chat_type == "group":
                    stmt = stmt.where(Message.group_id == chat_id)
                else:
                    stmt = stmt.where(Message.user_id == chat_id)
            result = await session.execute(stmt)
            return result.scalar_one()

    async def count_stickers(
        self, chat_type: str | None = None, chat_id: str | None = None
    ) -> int:
        async with self._session() as session:
            stmt = (
                select(func.count(Image.id))
                .join(Message, Image.message_id == Message.id)
                .where(Image.is_sticker)
            )
            if chat_type:
                stmt = stmt.where(Message.chat_type == chat_type)
            if chat_id:
                if chat_type == "group":
                    stmt = stmt.where(Message.group_id == chat_id)
                else:
                    stmt = stmt.where(Message.user_id == chat_id)
            result = await session.execute(stmt)
            return result.scalar_one()

    async def count_videos(
        self, chat_type: str | None = None, chat_id: str | None = None
    ) -> int:
        async with self._session() as session:
            stmt = select(func.count(Video.id)).join(
                Message, Video.message_id == Message.id
            )
            if chat_type:
                stmt = stmt.where(Message.chat_type == chat_type)
            if chat_id:
                if chat_type == "group":
                    stmt = stmt.where(Message.group_id == chat_id)
                else:
                    stmt = stmt.where(Message.user_id == chat_id)
            result = await session.execute(stmt)
            return result.scalar_one()

    async def count_downloaded_videos(
        self, chat_type: str | None = None, chat_id: str | None = None
    ) -> int:
        async with self._session() as session:
            stmt = (
                select(func.count(Video.id))
                .join(Message, Video.message_id == Message.id)
                .where(Video.downloaded)
            )
            if chat_type:
                stmt = stmt.where(Message.chat_type == chat_type)
            if chat_id:
                if chat_type == "group":
                    stmt = stmt.where(Message.group_id == chat_id)
                else:
                    stmt = stmt.where(Message.user_id == chat_id)
            result = await session.execute(stmt)
            return result.scalar_one()

    async def search_messages(
        self,
        keyword: str,
        chat_type: str | None = None,
        chat_id: str | None = None,
        limit: int = 10,
    ) -> list[Message]:
        async with self._session() as session:
            stmt = (
                select(Message)
                .options(
                    selectinload(Message.images),
                    selectinload(Message.videos),
                    selectinload(Message.at_mentions),
                    selectinload(Message.app_shares),
                )
                .where(Message.raw_message.contains(keyword))
                .order_by(desc(Message.timestamp))
            )
            if chat_type:
                stmt = stmt.where(Message.chat_type == chat_type)
            if chat_id:
                if chat_type == "group":
                    stmt = stmt.where(Message.group_id == chat_id)
                else:
                    stmt = stmt.where(Message.user_id == chat_id)
            stmt = stmt.limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def query_messages(
        self,
        *,
        user_id: str | None = None,
        chat_type: str | None = None,
        chat_id: str | None = None,
        keyword: str | None = None,
        time_from=None,
        time_to=None,
        has_forward: bool | None = None,
        has_image: bool | None = None,
        has_reply: bool | None = None,
        has_video: bool | None = None,
        has_at: bool | None = None,
        has_app_share: bool | None = None,
        limit: int = 20,
        order: str = "desc",
    ) -> list[Message]:
        async with self._session() as session:
            stmt = select(Message).options(
                selectinload(Message.images),
                selectinload(Message.videos),
                selectinload(Message.replies),
                selectinload(Message.forward_messages).options(
                    selectinload(ForwardMessage.children)
                ),
                selectinload(Message.at_mentions),
                selectinload(Message.app_shares),
            )
            stmt = _apply_message_query_filters(
                stmt,
                user_id=user_id,
                chat_type=chat_type,
                chat_id=chat_id,
                keyword=keyword,
                time_from=time_from,
                time_to=time_to,
                has_forward=has_forward,
                has_image=has_image,
                has_reply=has_reply,
                has_video=has_video,
                has_at=has_at,
                has_app_share=has_app_share,
            )

            if str(order or "desc").lower() == "asc":
                stmt = stmt.order_by(asc(Message.timestamp), asc(Message.id))
            else:
                stmt = stmt.order_by(desc(Message.timestamp), desc(Message.id))
            stmt = stmt.limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def backfill_forward_messages(
        self,
        message_db_id: int,
        forward_messages: list[dict],
    ) -> None:
        """Replace forward content for a message that was already stored."""
        async with self._session() as session:
            message = await session.get(Message, message_db_id)
            if message is None:
                return
            existing_stmt = select(ForwardMessage).where(
                ForwardMessage.message_id == message_db_id
            )
            existing = await session.execute(existing_stmt)
            for row in existing.scalars().all():
                await session.delete(row)

            if forward_messages:
                await _add_forward_messages(session, message_db_id, forward_messages)
                message.has_forward = True
            else:
                message.has_forward = False
            await session.commit()

    async def save_image(self, message_db_id: int, image_data: dict) -> Image:
        async def _save_once() -> Image:
            async with self._session() as session:
                try:
                    image = Image(
                        message_id=message_db_id,
                        file_url=image_data.get("file_url"),
                        file_unique=image_data.get("file_unique"),
                        file_size=image_data.get("file_size"),
                        local_path=image_data.get("local_path"),
                        width=image_data.get("width"),
                        height=image_data.get("height"),
                        downloaded=image_data.get("downloaded", False),
                    )
                    session.add(image)

                    msg_stmt = select(Message).where(Message.id == message_db_id)
                    msg_result = await session.execute(msg_stmt)
                    msg = msg_result.scalar_one()
                    msg.has_image = True

                    await session.commit()
                    await session.refresh(image)
                    return image
                except Exception:
                    await session.rollback()
                    raise

        return await self._run_with_lock_retry("save_image", _save_once)

    async def update_image_local_path(
        self, image_id: int, local_path: str, downloaded: bool = True
    ):
        async def _update_once() -> None:
            async with self._session() as session:
                try:
                    stmt = select(Image).where(Image.id == image_id)
                    result = await session.execute(stmt)
                    image = result.scalar_one()
                    image.local_path = local_path
                    image.downloaded = downloaded
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise

        await self._run_with_lock_retry("update_image_local_path", _update_once)

    async def find_reusable_image_by_url(self, file_url: str) -> Image | None:
        async with self._session() as session:
            stmt = (
                select(Image)
                .where(
                    Image.file_url == file_url,
                    Image.downloaded,
                    Image.local_path.is_not(None),
                )
                .order_by(desc(Image.id))
                .limit(1)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def apply_cached_image_to_message(
        self,
        message_db_id: int,
        file_url: str,
        local_path: str,
        file_unique: str | None,
        file_size: int | None,
        is_sticker: bool,
        sticker_confidence: float,
    ) -> None:
        async def _update_once() -> None:
            async with self._session() as session:
                try:
                    stmt = select(Image).where(
                        Image.message_id == message_db_id,
                        Image.file_url == file_url,
                    )
                    result = await session.execute(stmt)
                    image = result.scalar_one_or_none()
                    if image is None:
                        return
                    image.local_path = local_path
                    image.file_unique = file_unique
                    image.file_size = file_size
                    image.downloaded = True
                    image.is_sticker = is_sticker
                    image.sticker_confidence = sticker_confidence
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise

        await self._run_with_lock_retry("apply_cached_image_to_message", _update_once)

    async def find_reusable_video_by_url(self, file_url: str) -> Video | None:
        async with self._session() as session:
            stmt = (
                select(Video)
                .where(
                    Video.file_url == file_url,
                    Video.downloaded,
                    Video.local_path.is_not(None),
                )
                .order_by(desc(Video.id))
                .limit(1)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def apply_cached_video_to_message(
        self,
        message_db_id: int,
        *,
        file_url: str,
        local_path: str,
        file_unique: str | None,
        file_size: int | None,
    ) -> None:
        async def _update_once() -> None:
            async with self._session() as session:
                try:
                    stmt = select(Video).where(
                        Video.message_id == message_db_id,
                        Video.file_url == file_url,
                    )
                    result = await session.execute(stmt)
                    video = result.scalar_one_or_none()
                    if video is None:
                        return
                    video.local_path = local_path
                    video.file_unique = file_unique
                    video.file_size = file_size
                    video.downloaded = True
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise

        await self._run_with_lock_retry("apply_cached_video_to_message", _update_once)

    async def get_monitored_chats(self) -> list[MonitoredChat]:
        async with self._session() as session:
            stmt = select(MonitoredChat).where(MonitoredChat.enabled)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def add_monitored_chat(
        self, chat_type: str, chat_id: str, chat_name: str | None = None
    ) -> MonitoredChat:
        async with self._session() as session:
            try:
                existing_stmt = select(MonitoredChat).where(
                    MonitoredChat.chat_type == chat_type,
                    MonitoredChat.chat_id == chat_id,
                )
                existing_result = await session.execute(existing_stmt)
                existing = existing_result.scalar_one_or_none()
                if existing:
                    existing.enabled = True
                    existing.chat_name = chat_name or existing.chat_name
                    await session.commit()
                    await session.refresh(existing)
                    return existing

                chat = MonitoredChat(
                    chat_type=chat_type,
                    chat_id=chat_id,
                    chat_name=chat_name,
                    enabled=True,
                )
                session.add(chat)
                await session.commit()
                await session.refresh(chat)
                return chat
            except Exception:
                await session.rollback()
                raise

    async def remove_monitored_chat(self, chat_type: str, chat_id: str) -> bool:
        async with self._session() as session:
            try:
                stmt = select(MonitoredChat).where(
                    MonitoredChat.chat_type == chat_type,
                    MonitoredChat.chat_id == chat_id,
                )
                result = await session.execute(stmt)
                chat = result.scalar_one_or_none()
                if not chat:
                    return False
                chat.enabled = False
                await session.commit()
                return True
            except Exception:
                await session.rollback()
                raise

    async def is_chat_monitored(self, chat_type: str, chat_id: str) -> bool:
        async with self._session() as session:
            stmt = select(MonitoredChat).where(
                MonitoredChat.chat_type == chat_type,
                MonitoredChat.chat_id == chat_id,
                MonitoredChat.enabled,
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none() is not None

    # ── Image/Video analysis persistence ──────────────────────────────

    async def save_image_analysis(
        self,
        *,
        file_unique: str,
        model_used: str,
        analysis_json: str,
        media_type: str = "image",
        image_type: str = "",
        semantic_text: str = "",
        confidence: float = 0.0,
        prompt_version: str = "",
        schema_version: str = "",
        image_id: int | None = None,
        video_id: int | None = None,
        message_id: int | None = None,
    ) -> None:
        def _apply_row_update(row: ImageAnalysis) -> None:
            row.analysis_json = analysis_json
            row.media_type = media_type
            row.image_type = image_type
            row.semantic_text = semantic_text
            row.confidence = confidence
            row.prompt_version = prompt_version
            row.schema_version = schema_version
            if image_id is not None:
                row.image_id = image_id
            if video_id is not None:
                row.video_id = video_id
            if message_id is not None:
                row.message_id = message_id

        async def _upsert() -> None:
            async with self._session() as session:
                stmt = select(ImageAnalysis).where(
                    ImageAnalysis.file_unique == file_unique,
                    ImageAnalysis.model_used == model_used,
                )
                try:
                    result = await session.execute(stmt)
                    row = result.scalar_one_or_none()
                    if row is None:
                        session.add(
                            ImageAnalysis(
                                file_unique=file_unique,
                                model_used=model_used,
                                analysis_json=analysis_json,
                                media_type=media_type,
                                image_type=image_type,
                                semantic_text=semantic_text,
                                confidence=confidence,
                                prompt_version=prompt_version,
                                schema_version=schema_version,
                                image_id=image_id,
                                video_id=video_id,
                                message_id=message_id,
                            )
                        )
                    else:
                        _apply_row_update(row)
                    await session.commit()
                except IntegrityError:
                    await session.rollback()
                    result = await session.execute(stmt)
                    row = result.scalar_one_or_none()
                    if row is None:
                        raise
                    _apply_row_update(row)
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise

        await self._run_with_lock_retry("save_image_analysis", _upsert)

    async def get_image_analysis(
        self,
        file_unique: str,
        model_used: str | None = None,
    ) -> ImageAnalysis | None:
        async with self._session() as session:
            stmt = select(ImageAnalysis).where(ImageAnalysis.file_unique == file_unique)
            if model_used:
                stmt = stmt.where(ImageAnalysis.model_used == model_used)
            stmt = stmt.order_by(desc(ImageAnalysis.updated_at)).limit(1)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def get_image_analyses_by_message(
        self, message_db_id: int
    ) -> list[ImageAnalysis]:
        async with self._session() as session:
            stmt = (
                select(ImageAnalysis)
                .where(ImageAnalysis.message_id == message_db_id)
                .order_by(desc(ImageAnalysis.updated_at))
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())


def _ensure_message_sender_columns(sync_conn) -> None:
    result = sync_conn.exec_driver_sql("PRAGMA table_info(messages)")
    rows = list(result.fetchall())
    if not rows:
        return
    columns = {row[1] for row in rows}
    if "sender_nickname" not in columns:
        sync_conn.exec_driver_sql(
            "ALTER TABLE messages ADD COLUMN sender_nickname VARCHAR"
        )
    if "sender_card" not in columns:
        sync_conn.exec_driver_sql("ALTER TABLE messages ADD COLUMN sender_card VARCHAR")
    if "has_video" not in columns:
        sync_conn.exec_driver_sql(
            "ALTER TABLE messages ADD COLUMN has_video BOOLEAN DEFAULT FALSE"
        )

    analysis_rows = list(
        sync_conn.exec_driver_sql("PRAGMA table_info(image_analyses)").fetchall()
    )
    if analysis_rows:
        analysis_columns = {row[1] for row in analysis_rows}
        if "video_id" not in analysis_columns:
            sync_conn.exec_driver_sql(
                "ALTER TABLE image_analyses ADD COLUMN video_id INTEGER"
            )
        if "semantic_text" not in analysis_columns:
            sync_conn.exec_driver_sql(
                "ALTER TABLE image_analyses ADD COLUMN semantic_text TEXT DEFAULT ''"
            )
