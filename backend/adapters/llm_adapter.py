"""
LLM adapter: implements the LLMService port using ChatAnthropic.

Also contains generate_explanations(), which was split from the pure scorer
because it requires LLM interaction.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from measurement_design.ports import LLMService


class AnthropicLLMService:
    """Adapter implementing LLMService using ChatAnthropic."""

    def __init__(
        self,
        model: str = "claude-opus-4-5",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> None:
        self._llm = ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = await self._llm.ainvoke(messages)
        return response.content.strip()

    async def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        text = await self.generate_text(system_prompt, user_prompt)
        # Strip markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)
