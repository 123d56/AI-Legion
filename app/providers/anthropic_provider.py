"""Anthropic Claude 供应商适配器"""

from __future__ import annotations

from typing import Any

import anthropic

from app.models.schemas import Message
from app.providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        super().__init__(api_key, model, base_url)
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def _call(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        # Anthropic 需要把 system 消息单独提取
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

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt.strip():
            kwargs["system"] = system_prompt.strip()

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