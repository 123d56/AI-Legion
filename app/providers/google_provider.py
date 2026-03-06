"""Google Gemini 供应商适配器"""

from __future__ import annotations

from typing import Any, AsyncIterator

import google.generativeai as genai

from app.models.schemas import Message, StreamChunk
from app.providers.base import BaseProvider


class GoogleProvider(BaseProvider):
    name = "google"

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        super().__init__(api_key, model, base_url)
        genai.configure(api_key=api_key)
        self.gen_model = genai.GenerativeModel(model)

    def _prepare_history(self, messages: list[Message]) -> list[dict]:
        """将 messages 转换为 Gemini 格式"""
        history = []
        for m in messages:
            role = "user" if m.role in ("user", "system") else "model"
            history.append({"role": role, "parts": [m.content]})

        # Gemini 要求最后一条必须是 user
        if history and history[-1]["role"] != "user":
            history.append({"role": "user", "parts": ["请继续"]})

        return history

    async def _call(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        history = self._prepare_history(messages)

        chat = self.gen_model.start_chat(history=history[:-1])
        response = await chat.send_message_async(
            history[-1]["parts"][0],
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

        return {"content": response.text, "usage": None}

    async def _stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        history = self._prepare_history(messages)

        chat = self.gen_model.start_chat(history=history[:-1])
        response = await chat.send_message_async(
            history[-1]["parts"][0],
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
            stream=True,
        )

        async for chunk in response:
            if chunk.text:
                yield StreamChunk(
                    provider=self.name,
                    model=self.model,
                    delta=chunk.text,
                )