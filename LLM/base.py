from abc import ABC, abstractmethod
from typing import Any

from ..retry import RetryConfig
from ..schema import LLMResponse, Message

class LLMClientBase(ABC):
    def __init__(self,
                 api_key:str,
                 api_base:str,
                 model:str,
                 retry_config: RetryConfig | None = None,
                 ):
        self.api_key=api_key
        self.api_base=api_base
        self.model=model
        self.retry_config=retry_config or RetryConfig()
        self.retry_callback=None

    @abstractmethod
    async def generate(
        self,
        message:list[Message],
        tools:list[Any] | None=None,
    )->LLMResponse:
        pass

    @abstractmethod
    def _prepare_request(
        self,
        message:list[Message],
        tools:list[Any] | None=None,
    )->dict[str:Any]:
        pass

    @abstractmethod
    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        pass


