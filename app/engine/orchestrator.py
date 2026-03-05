"""
编排引擎 —— AI 军团的大脑
负责协调供应商和策略，产出最终回答
"""

from __future__ import annotations

import time

from app.models.schemas import ChatRequest, ChatResponse
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