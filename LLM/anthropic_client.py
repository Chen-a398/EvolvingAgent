import logging
from typing import Any

import anthropic

from ..retry import RetryConfig, async_retry
from ..schema import FunctionCall, LLMResponse, Message, TokenUsage, ToolCall
from .base import LLMClientBase

logger=logging.getLogger(__name__)

class AnthropicClient(LLMClientBase):
    def __init__(self,
                 api_key:str,
                 api_base:str="https://api.minimaxi.com/anthropic",
                 model:str= "MiniMax-M2.5",
                 retry_config: RetryConfig | None = None,
                 ):
                super().__init__(api_key, api_base, model, retry_config)
                self.client = anthropic.AsyncAnthropic(
            base_url=api_base,
            api_key=api_key,
            default_headers={"Authorization": f"Bearer {api_key}"},
        )
    async def _make_api_request(
        self,
        system_message: str | None,
        api_messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
    ) -> anthropic.types.Message:
        params = {
            "model": self.model,
            "max_tokens": 16384,
            "messages": api_messages,
        }

        if system_message:
            params["system"] = system_message

        if tools:
            params["tools"] = self._convert_tools(tools)

        # Use Anthropic SDK's async messages.create
        response = await self.client.messages.create(**params)
        return response  
    def _convert_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        result = []
        for tool in tools:
            if isinstance(tool, dict):
                result.append(tool)
            elif hasattr(tool, "to_schema"):
                # Tool object with to_schema method
                result.append(tool.to_schema())
            else:
                raise TypeError(f"Unsupported tool type: {type(tool)}")
        return result

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        system_message=None
        api_messages = []
        for msg in messages :
            if msg.role == "system":
                system_message=msg.content
                continue
        if msg.role in ["user","assistant"] :
            if msg.role=="assistant" and (msg.thinking or msg.tool_calls):
                content_blocks = []
                if msg.thinking:
                    content_blocks.append({"type": "thinking", "thinking": msg.thinking})
                if msg.content:
                    content_blocks.append({"type":"text","text":msg.content})    
                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "input": tool_call.function.arguments,
                        })
                api_messages.append({"role": "assistant", "content": content_blocks})
            else:
                api_messages.append({"role": msg.role, "content": msg.content})
        elif msg.role == "tool":
            api_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    }
                )
        return system_message,api_messages
    def _prepare_request(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        system_message, api_messages = self._convert_messages(messages)
        return {
            "system_message": system_message,
            "api_messages": api_messages,
            "tools": tools,
        }
    def _parse_response(self, response: anthropic.types.Message) -> LLMResponse:
        text_content = ""
        thinking_content = ""
        tool_calls = []
        for block in response.content:
            if block.type=="text":
                text_content+=block.text
            elif block.type == "thinking":
                thinking_content += block.thinking
            elif block.type == "tool_use":
                # Parse Anthropic tool_use block
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        type="function",
                        function=FunctionCall(
                            name=block.name,
                            arguments=block.input,
                        ),
                    )
                )  
        usage=None
        input_tokens = response.usage.input_tokens or 0
        output_tokens = response.usage.output_tokens or 0 
        cache_read_tokens = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        cache_creation_tokens = getattr(response.usage, "cache_creation_input_tokens", 0) or 0   
        total_input_tokens = input_tokens + cache_read_tokens + cache_creation_tokens
        usage = TokenUsage(
                prompt_tokens=total_input_tokens,
                completion_tokens=output_tokens,
                total_tokens=total_input_tokens + output_tokens,
            )          
        return LLMResponse(
            content=text_content,
            thinking=thinking_content if thinking_content else None,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=response.stop_reason or "stop",
            usage=usage,
        )    
    async def generate(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> LLMResponse:
            request_params = self._prepare_request(messages, tools)

        # Make API request with retry logic
            if self.retry_config.enabled:
                # Apply retry logic
                retry_decorator = async_retry(config=self.retry_config, on_retry=self.retry_callback)
                api_call = retry_decorator(self._make_api_request)
                response = await api_call(
                    request_params["system_message"],
                    request_params["api_messages"],
                    request_params["tools"],
                )
            else:
                # Don't use retry
                response = await self._make_api_request(
                    request_params["system_message"],
                    request_params["api_messages"],
                    request_params["tools"],
                )

            # Parse and return response
            return self._parse_response(response)


                        
                    







                
