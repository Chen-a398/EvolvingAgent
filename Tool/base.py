from typing import Any,List,Optional
from pydantic import BaseModel

class ToolResult(BaseModel):
    success:bool
    content:str
    error:str|None = None

class Tool(BaseModel):
    @property
    def name(self) -> str:
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def description(self) -> str:
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def parameters(self) -> dict[str,Any]:
        raise NotImplementedError("Subclasses must implement this method")

    @property
    async def execute(self, *args, **kwargs) -> ToolResult:
        raise NotImplementedError("Subclasses must implement this method")
    @property
    def to_schema(self) -> dict[str,Any]:
        return {
            "name": self.name(),
            "description": self.description(),
            "parameters": self.parameters(),
        }
    @property
    def to_openai_schema(self) -> dict[str,Any]:
        return {
            "type": "function",
            "function":{
                "name": self.name(),
                "description": self.description(),
                "parameters": self.parameters(),
            } 
        }


        

