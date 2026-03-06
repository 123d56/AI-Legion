"""流式响应测试"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.schemas import ChatRequest, Message, StreamChunk
from app.engine.orchestrator import Orchestrator
from app.providers.base import BaseProvider
from app.providers.registry import ProviderRegistry


class MockStreamProvider(BaseProvider):
    """模拟流式供应商"""

    name = "mock_stream"

    def __init__(self):
        super().__init__(api_key="test", model="mock-v1")
        self._chunks = ["你", "好", "，", "世", "界", "！"]

    async def _call(self, messages, temperature=0.7, max_tokens=4096):
        return {"content": "你好，世界！", "usage": None}

    async def _stream(self, messages, temperature=0.7, max_tokens=4096):
        for text in self._chunks:
            yield StreamChunk(
                provider=self.name,
                model=self.model,
                delta=text,
            )


class MockErrorProvider(BaseProvider):
    """模拟错误的供应商"""

    name = "mock_error"

    def __init__(self):
        super().__init__(api_key="test", model="error-v1")

    async def _call(self, messages, temperature=0.7, max_tokens=4096):
        raise Exception("模拟错误")

    async def _stream(self, messages, temperature=0.7, max_tokens=4096):
        raise Exception("流式模拟错误")
        yield  # noqa: 让它成为 async generator


def _parse_sse_events(sse_text: str) -> list[dict]:
    """解析 SSE 事件字符串为结构化数据"""
    events = []
    current_event = {}
    for line in sse_text.strip().split("\n"):
        if line.startswith("event: "):
            current_event["event"] = line[7:]
        elif line.startswith("data: "):
            current_event["data"] = json.loads(line[6:])
        elif line == "":
            if current_event:
                events.append(current_event)
                current_event = {}
    if current_event:
        events.append(current_event)
    return events


@pytest.mark.asyncio
async def test_stream_basic():
    """测试基本流式输出"""
    registry = ProviderRegistry()
    registry._providers["mock"] = MockStreamProvider()

    orch = Orchestrator(registry)
    request = ChatRequest(
        messages=[Message(role="user", content="你好")],
        stream=True,
    )

    events_raw = []
    async for event_str in orch.chat_stream(request):
        events_raw.append(event_str)

    # 拼接并解析所有事件
    full_sse = "".join(events_raw)
    events = _parse_sse_events(full_sse)

    # 应该有 6 个 chunk 事件 + 1 个 done 事件
    chunk_events = [e for e in events if e["event"] == "chunk"]
    done_events = [e for e in events if e["event"] == "done"]

    assert len(chunk_events) == 6
    assert len(done_events) == 1

    # 验证拼接后的完整文本
    full_text = "".join(e["data"]["delta"] for e in chunk_events)
    assert full_text == "你好，世界！"

    # 验证 done 事件包含延迟信息
    assert "latency_ms" in done_events[0]["data"]


@pytest.mark.asyncio
async def test_stream_fallback_on_error():
    """测试供应商报错时级联到下一个"""
    registry = ProviderRegistry()
    registry._providers["error"] = MockErrorProvider()
    registry._providers["mock"] = MockStreamProvider()

    orch = Orchestrator(registry)
    request = ChatRequest(
        messages=[Message(role="user", content="你好")],
        stream=True,
    )

    events_raw = []
    async for event_str in orch.chat_stream(request):
        events_raw.append(event_str)

    full_sse = "".join(events_raw)
    events = _parse_sse_events(full_sse)

    # 应该有错误事件和后续的 chunk 事件
    error_events = [e for e in events if e["event"] == "error"]
    chunk_events = [e for e in events if e["event"] == "chunk"]
    done_events = [e for e in events if e["event"] == "done"]

    assert len(error_events) >= 1  # 至少一个错误
    assert len(chunk_events) == 6  # 来自 mock 供应商
    assert len(done_events) == 1


@pytest.mark.asyncio
async def test_stream_no_providers():
    """测试无供应商时的流式输出"""
    registry = ProviderRegistry()
    orch = Orchestrator(registry)
    request = ChatRequest(
        messages=[Message(role="user", content="你好")],
        stream=True,
    )

    events_raw = []
    async for event_str in orch.chat_stream(request):
        events_raw.append(event_str)

    full_sse = "".join(events_raw)
    events = _parse_sse_events(full_sse)

    assert len(events) == 1
    assert events[0]["event"] == "error"
    assert "没有可用的模型供应商" in events[0]["data"]["message"]


@pytest.mark.asyncio
async def test_sse_event_format():
    """测试 SSE 事件格式正确性"""
    sse = Orchestrator._sse_event("chunk", {"delta": "你好"})
    assert sse.startswith("event: chunk\n")
    assert "data: " in sse
    assert sse.endswith("\n\n")

    # 解析 data
    data_line = sse.split("\n")[1]
    data = json.loads(data_line.replace("data: ", ""))
    assert data["delta"] == "你好"