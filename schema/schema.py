from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional,Any

class LLMProvider(str,Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class FunctionCall(BaseModel):
    name: str
    arguments: dict[str,Any]

class ToolCall(BaseModel):
    id: str
    type:str
    function: FunctionCall

class Message(BaseModel):
    role: str
    content:str| list[dict[str,Any]]
    thinking:str|None = None
    tool_calls:list[ToolCall]|None = None
    tool_call_id:str|None = None
    name:str|None = None

class TokenUsage(BaseModel):
    prompt_tokens:int=0
    completion_tokens:int=0
    total_tokens:int=0

class LLMResponse(BaseModel):
    content:str
    thinking:str|None = None
    tool_calls:list[ToolCall]|None = None
    finish_reason:str
    token_usage:TokenUsage | None = None
    





