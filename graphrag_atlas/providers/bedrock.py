from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from graphrag_atlas.providers.base import LLM, Embedder


class BedrockEmbedder(Embedder):
    """AWS Bedrock embedding provider (e.g. amazon.titan-embed-text-v2:0)."""

    def __init__(
        self,
        region: str = "us-east-1",
        model: str = "amazon.titan-embed-text-v2:0",
        dimension: int = 1024,
    ) -> None:
        self._region = region or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self._model = model
        self._dimension = dimension

        try:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=self._region)
        except ImportError:
            raise ImportError(
                "Package 'boto3' is required. Install via `pip install graphrag-atlas[aws]`."
            )

    @property
    def dimension(self) -> int:
        return self._dimension

    def _sync_embed(self, text: str) -> list[float]:
        body = json.dumps(
            {"inputText": text, "dimensions": self._dimension, "normalize": True}
        )
        response = self._client.invoke_model(
            modelId=self._model,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        data = json.loads(response["body"].read().decode("utf-8"))
        return data["embedding"]

    async def embed_text(self, text: str) -> list[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_embed, text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        tasks = [self.embed_text(t) for t in texts]
        return await asyncio.gather(*tasks)


class BedrockLLM(LLM):
    """AWS Bedrock language model provider (e.g. Amazon Nova Lite, Anthropic Claude)."""

    def __init__(
        self,
        region: str = "us-east-1",
        model: str = "us.amazon.nova-lite-v1:0",
        temperature: float = 0.0,
    ) -> None:
        self._region = region or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self._model = model
        self._temperature = temperature

        try:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=self._region)
        except ImportError:
            raise ImportError(
                "Package 'boto3' is required. Install via `pip install graphrag-atlas[aws]`."
            )

    def _sync_generate(self, prompt: str, schema: dict[str, Any] | None = None) -> str:
        if "nova" in self._model:
            body = {
                "inferenceConfig": {
                    "temperature": self._temperature,
                    "max_new_tokens": 2048,
                },
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
            }
        elif "claude" in self._model:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2048,
                "temperature": self._temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
        else:
            body = {"inputText": prompt}

        response = self._client.invoke_model(
            modelId=self._model,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        res_data = json.loads(response["body"].read().decode("utf-8"))

        if "nova" in self._model:
            return res_data["output"]["message"]["content"][0]["text"]
        elif "claude" in self._model:
            return res_data["content"][0]["text"]
        return json.dumps(res_data)

    async def generate(self, prompt: str, schema: dict[str, Any] | None = None) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_generate, prompt, schema)
