"""External-API generation backend (Claude / GPT-4o / OpenAI-compatible VLMs).

Sanctioned uses per the implementation report: offline rationale generation
(Stage 3) and the API policy VLM during dataset construction. NEVER use this
for the retrieval step — retrieval stays on local indices.

Key lookup: the environment variable named by ``extra.api_key_env``, falling
back to ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` per provider.

``GenerationConfig.extra`` knobs:
    base_url    — OpenAI-compatible endpoint (e.g. DashScope for Qwen-VL:
                  https://dashscope.aliyuncs.com/compatible-mode/v1), or a
                  LIST of endpoints — e.g. one local vLLM server per GPU —
                  which requests then rotate over (openai provider only).
                  With provider ``anthropic`` it points the client at any
                  Anthropic-protocol endpoint (one URL).
    api_key_env — env var holding the key (e.g. DASHSCOPE_API_KEY)

Multimodal: pass ``image_path=...`` to :meth:`generate` and the image is
attached base64-encoded (anthropic image block / openai image_url data URI).
Leave it None for text-only models.
"""

from __future__ import annotations

import base64
import itertools
import mimetypes
import os
import threading
import time
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

from .base import Generator, GenerationConfig

_DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-4o",
}
_DEFAULT_KEY_ENVS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


class APIGenerator(Generator):
    def __init__(self, config: GenerationConfig) -> None:
        super().__init__(config)
        self.provider = config.provider.lower()
        self.model = config.model or _DEFAULT_MODELS.get(self.provider)
        if self.model is None:
            raise ValueError(f"No default model for provider {self.provider!r}.")
        raw_url = config.extra.get("base_url")
        self.base_urls: list[str | None] = (
            [str(u) for u in raw_url] if isinstance(raw_url, (list, tuple)) else [raw_url]
        )
        if not self.base_urls:
            raise ValueError("base_url list is empty.")
        if len(self.base_urls) > 1 and self.provider != "openai":
            raise ValueError("A list of base_url endpoints needs provider 'openai'.")
        self.base_url = self.base_urls[0]
        self.api_key_env = (
            config.extra.get("api_key_env") or _DEFAULT_KEY_ENVS.get(self.provider, "")
        )
        self._clients = [self._make_client(url) for url in self.base_urls]
        self._client = self._clients[0]
        self._rotation = itertools.cycle(range(len(self._clients)))
        self._rotation_lock = threading.Lock()

    def _next_client(self):
        """Round-robin over the configured endpoints (thread-safe)."""
        if len(self._clients) == 1:
            return self._clients[0]
        with self._rotation_lock:
            return self._clients[next(self._rotation)]

    def _make_client(self, base_url: str | None = None):
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"{self.api_key_env} not set (provider={self.provider!r}). "
                "Fill it in DatasetConstruct/.env or export it."
            )
        timeout = float(self.config.extra.get("timeout", 120))
        max_retries = int(self.config.extra.get("max_retries", 2))
        if self.provider == "anthropic":
            import anthropic

            # base_url lets Anthropic-protocol endpoints of other vendors
            # (e.g. Moonshot's Kimi) stand in for api.anthropic.com.
            kwargs = {"base_url": base_url} if base_url else {}
            return anthropic.Anthropic(
                api_key=api_key, timeout=timeout, max_retries=max_retries, **kwargs
            )
        if self.provider == "openai":
            import httpx
            import openai

            kwargs: dict[str, Any] = {}
            if _is_local_base_url(base_url):
                # Local OpenAI-compatible servers such as Qwen run on
                # 127.0.0.1. Environment proxies can otherwise intercept the
                # request and return a misleading 502.
                kwargs["http_client"] = httpx.Client(
                    trust_env=False,
                    timeout=timeout,
                )
            return openai.OpenAI(
                api_key=api_key, base_url=base_url,
                timeout=timeout, max_retries=max_retries,
                **kwargs,
            )
        raise ValueError(f"Unknown provider {self.provider!r}.")

    # ------------------------------------------------------------------ #
    def generate(
        self, question: str, context_docs: Sequence[str], **kwargs: Any
    ) -> str:
        attempts = int(self.config.extra.get("request_attempts", 3))
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                return self._generate_once(question, context_docs, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt + 1 >= attempts:
                    break
                time.sleep(min(2.0 * (attempt + 1), 6.0))
        assert last_exc is not None
        raise last_exc

    def _generate_once(
        self, question: str, context_docs: Sequence[str], **kwargs: Any
    ) -> str:
        user_text = self._build_prompt(question, context_docs)
        image_paths = _image_paths(kwargs)
        if self.provider == "anthropic":
            content: list[dict] = []
            for image_path in image_paths:
                media_type, data = _read_image_b64(image_path)
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": data},
                })
            content.append({"type": "text", "text": user_text})
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                system=self.config.system_prompt,
                messages=[{"role": "user", "content": content}],
            )
            return resp.content[0].text.strip()

        # openai / OpenAI-compatible
        if image_paths:
            user_content: Any = []
            for image_path in image_paths:
                media_type, data = _read_image_b64(image_path)
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{data}"},
                    }
                )
            user_content.append({"type": "text", "text": user_text})
        else:
            user_content = user_text
        seed = self.config.extra.get("seed")
        resp = self._next_client().chat.completions.create(
            model=self.model,
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": user_content},
            ],
            # Sampling seed: with temperature > 0 this is what makes a run
            # reproducible, and what separates the 0/1/2 seed replicates.
            # vLLM and the OpenAI API both honour it; providers that do not
            # simply ignore the field.
            **({"seed": int(seed)} if seed is not None else {}),
            # Provider-specific passthrough, e.g. DeepSeek reasoning models
            # need {"thinking": {"type": "disabled"}} or thinking tokens eat
            # the whole max_tokens budget and content comes back empty.
            extra_body=self.config.extra.get("extra_body"),
        )
        return (resp.choices[0].message.content or "").strip()


def _read_image_b64(path: str) -> tuple[str, str]:
    media_type = mimetypes.guess_type(path)[0] or "image/jpeg"
    data = base64.standard_b64encode(Path(path).read_bytes()).decode("ascii")
    return media_type, data


def _image_paths(kwargs: dict[str, Any]) -> list[str]:
    raw = kwargs.get("image_paths")
    paths = list(raw) if raw else []
    single = kwargs.get("image_path")
    if single:
        paths.insert(0, single)
    limit = int(kwargs.get("max_images", 4))
    return list(dict.fromkeys(str(path) for path in paths if path))[:limit]


def _is_local_base_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    host = (urlparse(base_url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}
