"""DeepSeek 供应商适配器 (兼容 OpenAI API 格式)"""

from __future__ import annotations

from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from app.models.schemas import Message, StreamChunk
from app.providers.base import BaseProvider


class DeepSeekProvider(BaseProvider):
    """
    DeepSeek 使用兼容 OpenAI 的 API 格式，
    通过设置 base_url 指向 DeepSeek 的端点。
    """

    name = "deepseek"

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        super().__init__(api_key, model, base_url)
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or "https://api.deepseek.com",
        )

    async def _call(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return {"content": choice.message.content or "", "usage": usage}

    async def _stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield StreamChunk(
                    provider=self.name,
                    model=self.model,
                    delta=chunk.choices[0].delta.content,
                )