"""
OpenRouter client for LLM inference.

Provides a simple async/sync client for making inference requests
to OpenRouter, which provides access to 400+ models via a single API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, ClassVar

import httpx


@dataclass
class Message:
    """A chat message."""

    role: str  # "system", "user", or "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to API-compatible dict."""
        return {"role": self.role, "content": self.content}


@dataclass
class CompletionResponse:
    """Response from a completion request."""

    content: str
    model: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    raw_response: dict[str, Any] = field(default_factory=dict)


class OpenRouterClient:
    """
    Client for OpenRouter API.

    OpenRouter provides access to 400+ models from multiple providers
    (OpenAI, Anthropic, Meta, Mistral, etc.) via a single API endpoint.

    Example:
        >>> client = OpenRouterClient()
        >>> response = client.complete(
        ...     messages=[Message("user", "Write a haiku about coding")],
        ...     model="meta-llama/llama-3.1-8b-instruct"
        ... )
        >>> print(response.content)

    Environment Variables:
        OPENROUTER_API_KEY: Your OpenRouter API key (required)
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    # Default model for testing (free tier, good for poetry)
    DEFAULT_MODEL = "deepseek/deepseek-chat-v3-0324:free"

    # Models for poetry generation (with :free suffix for free tier)
    # See https://openrouter.ai/models for current availability
    # Last updated: December 2025
    RECOMMENDED_MODELS: ClassVar[dict[str, str]] = {
        # Free tier models (rate limited, for development/testing)
        "free-fast": "google/gemma-3-27b-it:free",
        "free-reasoning": "qwen/qwq-32b:free",
        "free-quality": "deepseek/deepseek-chat-v3-0324:free",
        "free-thinking": "google/gemini-2.0-flash-thinking-exp:free",
        # Paid models (no rate limits, production use)
        "paid-fast": "google/gemini-2.0-flash-exp",
        "paid-quality": "anthropic/claude-3-haiku",
    }

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        app_name: str = "abide-evals",
        timeout: float = 60.0,
    ) -> None:
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            base_url: Override base URL (for testing)
            app_name: Name of your app (shows in OpenRouter dashboard)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.base_url = base_url or self.BASE_URL
        self.app_name = app_name
        self.timeout = timeout

        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/darrenangle/abide",
            "X-Title": self.app_name,
            "Content-Type": "application/json",
        }

    def _get_client(self) -> httpx.Client:
        """Get or create sync HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                headers=self._get_headers(),
                timeout=self.timeout,
            )
        return self._client

    def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._get_headers(),
                timeout=self.timeout,
            )
        return self._async_client

    def close(self) -> None:
        """Close HTTP clients."""
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        """Close async HTTP client."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def __enter__(self) -> OpenRouterClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    async def __aenter__(self) -> OpenRouterClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    def complete(
        self,
        messages: list[Message] | list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Make a synchronous completion request.

        Args:
            messages: List of messages (Message objects or dicts)
            model: Model ID (e.g., "meta-llama/llama-3.1-8b-instruct")
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional API parameters

        Returns:
            CompletionResponse with generated content
        """
        model = model or self.DEFAULT_MODEL

        # Convert messages to dicts
        msg_dicts = [m.to_dict() if isinstance(m, Message) else m for m in messages]

        payload = {
            "model": model,
            "messages": msg_dicts,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        client = self._get_client()
        response = client.post("/chat/completions", json=payload)
        response.raise_for_status()

        return self._parse_response(response.json())

    async def acomplete(
        self,
        messages: list[Message] | list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Make an asynchronous completion request.

        Args:
            messages: List of messages (Message objects or dicts)
            model: Model ID (e.g., "meta-llama/llama-3.1-8b-instruct")
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional API parameters

        Returns:
            CompletionResponse with generated content
        """
        model = model or self.DEFAULT_MODEL

        # Convert messages to dicts
        msg_dicts = [m.to_dict() if isinstance(m, Message) else m for m in messages]

        payload = {
            "model": model,
            "messages": msg_dicts,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        client = self._get_async_client()
        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()

        return self._parse_response(response.json())

    def _parse_response(self, data: dict[str, Any]) -> CompletionResponse:
        """Parse API response into CompletionResponse."""
        choice = data["choices"][0]
        usage = data.get("usage", {})

        return CompletionResponse(
            content=choice["message"]["content"],
            model=data.get("model", "unknown"),
            finish_reason=choice.get("finish_reason", "unknown"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            raw_response=data,
        )

    def generate_poem(
        self,
        form_instruction: str,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.8,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate a poem based on form instructions.

        Args:
            form_instruction: Instructions for the poetic form
            model: Model to use (defaults to DEFAULT_MODEL)
            system_prompt: Optional system prompt override
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Generated poem text
        """
        if system_prompt is None:
            system_prompt = (
                "You are an expert poet. Generate poems that strictly follow "
                "the requested form and structure. Output ONLY the poem, with "
                "no additional commentary or explanation."
            )

        messages = [
            Message("system", system_prompt),
            Message("user", form_instruction),
        ]

        response = self.complete(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.content.strip()

    async def agenerate_poem(
        self,
        form_instruction: str,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.8,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate a poem asynchronously.

        Args:
            form_instruction: Instructions for the poetic form
            model: Model to use (defaults to DEFAULT_MODEL)
            system_prompt: Optional system prompt override
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Generated poem text
        """
        if system_prompt is None:
            system_prompt = (
                "You are an expert poet. Generate poems that strictly follow "
                "the requested form and structure. Output ONLY the poem, with "
                "no additional commentary or explanation."
            )

        messages = [
            Message("system", system_prompt),
            Message("user", form_instruction),
        ]

        response = await self.acomplete(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.content.strip()
