import httpx
import asyncio
import logging
import json
from typing import Dict, Any, Optional, Literal
from ..config import settings

logger = logging.getLogger(__name__)

class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    pass

class LLMClient:
    def __init__(
        self,
        provider: Optional[Literal["openai", "gemini"]] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = settings.LLM_TEMPERATURE,
        timeout: int = settings.LLM_TIMEOUT,
        max_retries: int = settings.LLM_MAX_RETRIES,
    ):
        self.provider = provider or settings.LLM_PROVIDER
        self.api_key = api_key or (settings.OPENAI_API_KEY if self.provider == "openai" else settings.GEMINI_API_KEY)
        self.model = model or (settings.OPENAI_MODEL if self.provider == "openai" else settings.GEMINI_MODEL)
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            raise LLMClientError(f"API key missing for provider: {self.provider}")

    async def classify_segment(self, *, segment_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sends segment text to LLM and returns parsed JSON dict."""
        if self.provider == "openai":
            return await self._call_openai(segment_text, metadata, json_mode=True)
        elif self.provider == "gemini":
            return await self._call_gemini(segment_text, metadata, json_mode=True)
        else:
            raise LLMClientError(f"Unsupported provider: {self.provider}")

    async def generate_text(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """Sends a prompt to LLM and returns the raw text response."""
        metadata = {"system_prompt": system_prompt}
        if self.provider == "openai":
            # For text generation, we call _call_openai with json_mode=False
            result = await self._call_openai(prompt, metadata, json_mode=False)
            return result.get("text", "")
        elif self.provider == "gemini":
            result = await self._call_gemini(prompt, metadata, json_mode=False)
            return result.get("text", "")
        else:
            raise LLMClientError(f"Unsupported provider: {self.provider}")

    async def _call_openai(self, segment_text: str, metadata: Dict[str, Any], json_mode: bool = True) -> Dict[str, Any]:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": metadata.get("system_prompt", "You are a helpful assistant.")},
                {"role": "user", "content": segment_text}
            ],
            "temperature": self.temperature,
        }
        
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        response_data = await self._request_with_retry(url, headers, payload)
        try:
            content = response_data["choices"][0]["message"]["content"]
            if json_mode:
                return json.loads(content)
            else:
                return {"text": content}
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse OpenAI response: {e}")
            raise LLMClientError(f"Invalid OpenAI response structure: {e}")

    async def _call_gemini(self, segment_text: str, metadata: Dict[str, Any], json_mode: bool = True) -> Dict[str, Any]:
        # Gemini implementation using REST API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        system_instruction = metadata.get("system_prompt", "You are a helpful assistant.")
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"SYSTEM INSTRUCTION: {system_instruction}\n\nUSER INPUT: {segment_text}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
            }
        }
        
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        response_data = await self._request_with_retry(url, headers, payload)
        
        try:
            # Parse Gemini's specific response structure
            content = response_data["candidates"][0]["content"]["parts"][0]["text"]
            if json_mode:
                return json.loads(content)
            else:
                return {"text": content}
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            raise LLMClientError(f"Invalid Gemini response structure: {e}")

    async def _request_with_retry(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    
                    if response.status_code == 200:
                        return response.json()
                    
                    # Handle specific error codes
                    if response.status_code == 429:
                        logger.warning(f"Rate limited (429). Attempt {attempt + 1}/{self.max_retries + 1}")
                    elif response.status_code >= 500:
                        logger.warning(f"Server error ({response.status_code}). Attempt {attempt + 1}/{self.max_retries + 1}")
                    else:
                        # 4xx errors other than 429 should not be retried
                        logger.error(f"LLM API Error: {response.status_code} - {response.text}")
                        raise LLMClientError(f"API Error {response.status_code}: {response.text}")
                    
                    response.raise_for_status()

            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = (2 ** attempt)  # Simple exponential backoff
                    await asyncio.sleep(wait_time)
                continue
        
        raise LLMClientError(f"Failed after {self.max_retries + 1} attempts. Last error: {last_exception}")
