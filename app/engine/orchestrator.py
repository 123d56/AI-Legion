"""
编排引擎 —— AI 军团的大脑
负责协调供应商和策略，产出最终回答
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    Message,
    StreamChunk,
)
from app.providers.base import BaseProvider
from app.providers.registry import ProviderRegistry
from app.engine.strategies import get_strategy
from config.settings import settings


class Orchestrator:
    """编排引擎"""

    def __init__(self, registry: ProviderRegistry) -> None:
        self.registry = registry

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """处理一次聊天请求"""
        providers = self.registry.get_all()

        if not providers:
            return ChatResponse(
                answer="❌ 没有可用的模型供应商，请在 .env 中配置至少一个 API 密钥。",
                strategy_used="none",
                provider_responses=[],
                total_latency_ms=0,
            )

        # 如果只有一个供应商，直接调用，不走聚合
        strategy_name = request.strategy or settings.strategy

        if len(providers) == 1:
            strategy_name = "cascade"  # 单供应商时退化为直连

        strategy = get_strategy(strategy_name)

        start = time.perf_counter()
        answer, provider_responses = await strategy.execute(
            providers=providers,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        total_ms = round((time.perf_counter() - start) * 1000, 2)

        return ChatResponse(
            answer=answer,
            strategy_used=strategy_name,
            provider_responses=provider_responses,
            total_latency_ms=total_ms,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """
        流式聊天 —— 产出 SSE 格式的事件流

        流式模式下使用「级联」策略：选择第一个可用的供应商进行流式输出。
        多供应商并行流式过于复杂，且用户体验不佳（文本交叉），
        因此流式模式下退化为单供应商流式。

        SSE 事件格式：
            event: chunk
            data: {"provider": "openai", "model": "gpt-4o", "delta": "你好"}

            event: done
            data: {"provider": "openai", "model": "gpt-4o", "latency_ms": 1234.56}

            event: error
            data: {"message": "..."}
        """
        providers = self.registry.get_all()

        if not providers:
            yield self._sse_event("error", {
                "message": "❌ 没有可用的模型供应商，请在 .env 中配置至少一个 API 密钥。"
            })
            return

        # 流式模式：级联尝试每个供应商
        for provider in providers:
            start = time.perf_counter()
            has_output = False
            error_occurred = False

            try:
                async for chunk in provider.chat_stream(
                    messages=request.messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                ):
                    # 检查是否是错误信息
                    if chunk.delta.startswith("[ERROR]"):
                        error_occurred = True
                        yield self._sse_event("error", {
                            "provider": provider.name,
                            "model": provider.model,
                            "message": chunk.delta,
                        })
                        break

                    has_output = True
                    yield self._sse_event("chunk", {
                        "provider": provider.name,
                        "model": provider.model,
                        "delta": chunk.delta,
                    })

                if has_output and not error_occurred:
                    elapsed = round((time.perf_counter() - start) * 1000, 2)
                    yield self._sse_event("done", {
                        "provider": provider.name,
                        "model": provider.model,
                        "latency_ms": elapsed,
                    })
                    return  # 成功，结束流

            except Exception as exc:
                yield self._sse_event("error", {
                    "provider": provider.name,
                    "model": provider.model,
                    "message": str(exc),
                })
                continue  # 尝试下一个供应商

        # 所有供应商都失败了
        if not has_output:
            yield self._sse_event("error", {
                "message": "所有模型均返回错误，请检查 API 配置。"
            })

    @staticmethod
    def _sse_event(event: str, data: dict) -> str:
        """格式化为 SSE 事件字符串"""
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"