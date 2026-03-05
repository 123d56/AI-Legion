"""
模型供应商的抽象基类
所有供应商适配器必须继承此类
"""

from __future__ import annotations

import abc
import time
from typing import Any

from app.models.schemas import Message, ProviderResponse


class BaseProvider(abc.ABC):
    """模型供应商抽象基类"""

    name: str = "base"

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    @abc.abstractmethod
    async def _call(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """
        调用模型 API，返回原始结果。
        子类必须实现此方法。

        Returns:
            {"content": str, "usage": dict | None}
        """
        ...

    async def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ProviderResponse:
        """统一的调用入口，自动计时和异常处理"""
        start = time.perf_counter()
        try:
            result = await self._call(messages, temperature, max_tokens)
            elapsed = (time.perf_counter() - start) * 1000
            return ProviderResponse(
                provider=self.name,
                model=self.model,
                content=result["content"],
                usage=result.get("usage"),
                latency_ms=round(elapsed, 2),
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return ProviderResponse(
                provider=self.name,
                model=self.model,
                content="",
                latency_ms=round(elapsed, 2),
                error=str(exc),
            )