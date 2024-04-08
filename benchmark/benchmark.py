from pydantic import Field, BaseModel
# from pydantic.json import pydantic_encoder
from typing import Optional, List, Dict, Any
from openai.types.chat import ChatCompletion


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]  # Adjust the type as needed

class ExpectedFunctionCall(BaseModel):
    name: str
    arguments: Dict[str, Any] 
    result: str

class Actual(BaseModel):
    tools: Optional[List[Any]] = []
    messages: Optional[List[Any]] = []
    responses: Optional[List[ChatCompletion]] = []

class TestCase(BaseModel):
    __test__ = False
    prompt: str
    available_functions: List[FunctionDefinition]
    expected_function_calls: List[ExpectedFunctionCall]
    actual: Actual = Field(default_factory=Actual)