"""Anthropic Claude 供应商适配器"""

from __future__ import annotations

from typing import Any, AsyncIterator

import anthropic

from app.models.schemas import Message, StreamChunk
from app.providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        super().__init__(api_key, model, base_url)
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    def _prepare_messages(self, messages: list[Message]) -> tuple[str, list[dict]]:
        """提取 system 消息和 chat 消息"""
        system_prompt = ""
        chat_messages = []
        for m in messages:
            if m.role == "system":
                system_prompt += m.content + "\n"
            else:
                chat_messages.append({"role": m.role, "content": m.content})

        # 确保至少有一条用户消息
        if not chat_messages:
            chat_messages = [{"role": "user", "content": "Hello"}]

        return system_prompt.strip(), chat_messages

    async def _call(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        system_prompt, chat_messages = self._prepare_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await self.client.messages.create(**kwargs)

        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }
        return {"content": content, "usage": usage}

    async def _stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        system_prompt, chat_messages = self._prepare_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        async with self.client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield StreamChunk(
                    provider=self.name,
                    model=self.model,
                    delta=text,
                )