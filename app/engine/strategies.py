"""
聚合策略模块
定义多种策略将多个模型的回答聚合成一个最终答案
"""

from __future__ import annotations

import abc
import asyncio
from typing import Any

from app.models.schemas import Message, ProviderResponse
from app.providers.base import BaseProvider


class BaseStrategy(abc.ABC):
    """聚合策略抽象基类"""

    name: str = "base"

    @abc.abstractmethod
    async def execute(
        self,
        providers: list[BaseProvider],
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[str, list[ProviderResponse]]:
        """
        执行聚合策略

        Returns:
            (最终回答, 各供应商原始响应列表)
        """
        ...


class VoteStrategy(BaseStrategy):
    """
    投票策��：并行调用所有模型，选取最长的回答作为最终答案。
    (后续版本会引入语义相似度投票)
    """

    name = "vote"

    async def execute(
        self,
        providers: list[BaseProvider],
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[str, list[ProviderResponse]]:
        # 并行调用所有供应商
        tasks = [p.chat(messages, temperature, max_tokens) for p in providers]
        responses: list[ProviderResponse] = await asyncio.gather(*tasks)

        # 过滤掉出错的响应
        valid = [r for r in responses if not r.error and r.content.strip()]

        if not valid:
            return "所有模型均返回错误，请检查 API 配置。", responses

        # 简单策略：选最长的回答（信息量最大）
        best = max(valid, key=lambda r: len(r.content))
        return best.content, responses


class BestOfStrategy(BaseStrategy):
    """
    择优策略：并行调用所有模型，选取响应最快且无错误的回答。
    """

    name = "best_of"

    async def execute(
        self,
        providers: list[BaseProvider],
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[str, list[ProviderResponse]]:
        tasks = [p.chat(messages, temperature, max_tokens) for p in providers]
        responses: list[ProviderResponse] = await asyncio.gather(*tasks)

        valid = [r for r in responses if not r.error and r.content.strip()]

        if not valid:
            return "所有模型均返回错误，请检查 API 配置。", responses

        # 选最快的
        fastest = min(valid, key=lambda r: r.latency_ms)
        return fastest.content, responses


class CascadeStrategy(BaseStrategy):
    """
    级联策略：按供应商顺序逐个尝试，第一个成功的即返回。
    适合用于故障转移场景。
    """

    name = "cascade"

    async def execute(
        self,
        providers: list[BaseProvider],
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[str, list[ProviderResponse]]:
        responses: list[ProviderResponse] = []

        for provider in providers:
            resp = await provider.chat(messages, temperature, max_tokens)
            responses.append(resp)
            if not resp.error and resp.content.strip():
                return resp.content, responses

        return "所有模型均返回错误，请检查 API 配置。", responses


class ChainStrategy(BaseStrategy):
    """
    链式策略：第一个模型回答后，将回答作为上下文传给下一个模型，
    让其改进/补充，层层优化。
    """

    name = "chain"

    async def execute(
        self,
        providers: list[BaseProvider],
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[str, list[ProviderResponse]]:
        responses: list[ProviderResponse] = []
        current_messages = list(messages)

        for i, provider in enumerate(providers):
            resp = await provider.chat(current_messages, temperature, max_tokens)
            responses.append(resp)

            if resp.error or not resp.content.strip():
                continue

            if i < len(providers) - 1:
                # 将当前回答加入上下文，让下一个模型改进
                current_messages = list(messages) + [
                    Message(role="assistant", content=resp.content),
                    Message(
                        role="user",
                        content="请在以上回答的基础上进行改进、补充和完善，给出更好的回答。",
                    ),
                ]

        # 最终答案是最后一个成功响应
        valid = [r for r in responses if not r.error and r.content.strip()]
        if valid:
            return valid[-1].content, responses
        return "所有模型均返回错误，请检查 API 配置。", responses


# 策略注册表
STRATEGIES: dict[str, type[BaseStrategy]] = {
    "vote": VoteStrategy,
    "best_of": BestOfStrategy,
    "cascade": CascadeStrategy,
    "chain": ChainStrategy,
}


def get_strategy(name: str) -> BaseStrategy:
    """根据名称获取策略实例"""
    cls = STRATEGIES.get(name)
    if cls is None:
        raise ValueError(f"未知策略: {name}，可选: {list(STRATEGIES.keys())}")
    return cls()