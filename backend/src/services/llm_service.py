import asyncio
import httpx
import logging
from typing import List, Dict, Any, Optional
from ..config import settings
from ..models.chat import Message, ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice, ChatCompletionUsage, Source

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service to interface with OpenRouter for LLM completions
    with fallback mechanisms
    """

    def __init__(self):
        self.api_key = settings.openrouter_api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.timeout = httpx.Timeout(60.0, connect=10.0)
        self.primary_model = settings.primary_model
        self.fallback_model_1 = settings.fallback_model_1
        self.fallback_model_2 = settings.fallback_model_2

    async def get_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> ChatCompletionResponse:
        """
        Get completion from LLM with fallback mechanisms
        """
        if model is None:
            model = self.primary_model

        models_to_try = [model]
        if model != self.primary_model:
            models_to_try = [model, self.primary_model, self.fallback_model_1, self.fallback_model_2]
        else:
            models_to_try = [self.primary_model, self.fallback_model_1, self.fallback_model_2]

        last_error = None

        for attempt_model in models_to_try:
            try:
                response = await self._call_llm(
                    messages=messages,
                    model=attempt_model,
                    max_tokens=max_tokens or settings.max_tokens,
                    temperature=temperature or settings.temperature,
                    top_p=top_p or settings.top_p
                )
                logger.info(f"Successfully got completion using model: {attempt_model}")
                return response
            except Exception as e:
                logger.warning(f"Failed to get completion from model {attempt_model}: {str(e)}")
                last_error = e
                continue

        # If all models failed, raise the last error
        raise last_error or Exception("All LLM models failed to respond")

    async def _call_llm(
        self,
        messages: List[Message],
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float
    ) -> ChatCompletionResponse:
        """
        Make a single call to the LLM
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Convert messages to the format expected by OpenRouter
        formatted_messages = [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]

        payload = {
            "model": model,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False  # We'll handle streaming separately if needed
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )

            if response.status_code != 200:
                raise Exception(f"LLM API error: {response.status_code} - {response.text}")

            data = response.json()

            # Convert the response to our ChatCompletionResponse format
            choices = []
            for choice in data["choices"]:
                message = Message(
                    role=choice["message"]["role"],
                    content=choice["message"]["content"]
                )
                choices.append(
                    ChatCompletionChoice(
                        index=choice["index"],
                        message=message,
                        finish_reason=choice["finish_reason"]
                    )
                )

            usage = ChatCompletionUsage(
                prompt_tokens=data["usage"]["prompt_tokens"],
                completion_tokens=data["usage"]["completion_tokens"],
                total_tokens=data["usage"]["total_tokens"]
            )

            # Create a mock response ID and timestamp
            import time
            response_id = f"chatcmpl-{int(time.time())}"
            created_timestamp = int(time.time())

            return ChatCompletionResponse(
                id=response_id,
                created=created_timestamp,
                model=model,
                conversation_id="",  # Will be set by the calling service
                choices=choices,
                usage=usage
            )

    async def generate_embedding(self, text: str, model: str = "text-embedding-ada-002") -> List[float]:
        """
        Generate embedding for text.
        In this implementation, we delegate to the vector store service
        which has its own embedding generation method.
        """
        # For now, we'll use the vector store's embedding generation
        # In a real implementation, you might want to use OpenAI's embedding API directly
        from .vector_store import vector_store_service
        return await vector_store_service.generate_embedding(text)


# Global instance
llm_service = LLMService()