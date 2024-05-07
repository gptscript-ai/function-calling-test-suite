import json
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
    optional: Optional[bool] = False


class ActualFunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None


class Actual(BaseModel):
    tools: Optional[List[Dict[str, Any]]] = []
    messages: Optional[List[Dict[str, Any]]] = []
    function_calls: Optional[List[ActualFunctionCall]] = []
    responses: Optional[List[Dict[str, Any]]] = []
    answers: Optional[List[str]] = []
    judge_ruling: Optional[str] = None

class TestCase(BaseModel):
    __test__ = False

    description: Optional[str] = None
    categories: Set[str]

    prompt: str
    available_functions: List[FunctionDefinition]
    expected_function_calls: List[ExpectedFunctionCall]
    final_answer_should: Optional[str] = None

    actual: Actual = Field(default_factory=Actual)

    @field_validator('categories')
    @classmethod
    def ensure_categories(cls, categories):
        if len(categories) < 1:
            raise ValueError("Must contain at least one category")

        return categories

    @classmethod
    def create(cls, description, categories, prompt, functions, function_calls):
        return cls(
            description=description,
            categories=set(categories),
            prompt=prompt,
            available_functions=[FunctionDefinition(**func) for func in functions],
            expected_function_calls=[ExpectedFunctionCall(**call) for call in function_calls]
        )
