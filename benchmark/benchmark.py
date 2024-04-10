from pydantic import Field, BaseModel, field_validator
from typing import Optional, Set, List, Dict, Any
from openai.types.chat import ChatCompletion


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]  # Adjust the type as needed

class ExpectedFunctionCall(BaseModel):
    name: Optional[str] = None 
    arguments: Optional[Dict[str, Any]] = None 
    result: Optional[str] = None 
    finish_reason: Optional[str] = "tool_calls"

class Actual(BaseModel):
    tools: Optional[List[Any]] = []
    messages: Optional[List[Any]] = []
    responses: Optional[List[ChatCompletion]] = []

class TestCase(BaseModel):
    __test__ = False

    description: Optional[str] = None
    categories: Set[str]

    prompt: str
    available_functions: List[FunctionDefinition]
    expected_function_calls: List[ExpectedFunctionCall]
    actual: Actual = Field(default_factory=Actual)

    @field_validator('categories')
    @classmethod
    def ensure_categories(cls, categories):
        if len(categories) < 1:
            raise ValueError("Must contain at least one category")

        return categories

    @field_validator('expected_function_calls')
    @classmethod
    def ensure_expected_function_calls(cls, expected_function_calls):
        if len(expected_function_calls) < 1:
            raise ValueError("Must contain at least one expected call")
        
        stop_index = -1 
        for index, expected_call in enumerate(expected_function_calls):
            if stop_index >= 0: 
                raise ValueError(f"Call [{stop_index}] with finish_reason=\"stop\" must be the final expected call")
            
            if expected_call.finish_reason == "stop":
                stop_index = index

        return expected_function_calls 