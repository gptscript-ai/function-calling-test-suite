from pydantic import Field, BaseModel, field_validator
from typing import Optional, Union, Set, List, Dict, Any


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]  # Adjust the type as needed


class ExpectedFunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    result: Optional[str] = None
    optional: Optional[bool] = False


class ExpectedFunctionCallGroup(BaseModel):
    any_order: Optional[List[ExpectedFunctionCall]]


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

    system_prompt: Optional[str] = None
    prompt: str
    available_functions: List[FunctionDefinition]
    expected_function_calls: List[Union[ExpectedFunctionCall, ExpectedFunctionCallGroup]]
    final_answer_should: Optional[str] = None

    actual: Actual = Field(default_factory=Actual)

    @field_validator('categories')
    @classmethod
    def ensure_categories(cls, categories):
        if len(categories) < 1:
            raise ValueError("Must contain at least one category")

        return categories

    @classmethod
    def parse_json(cls, obj: Dict[str, Any]):
        # Preprocess expected_function_calls to handle mixed content
        function_calls = []
        for call in obj['expected_function_calls']:
            if 'any_order' in call:
                group = ExpectedFunctionCallGroup(any_order=[ExpectedFunctionCall(**f) for f in call['any_order']])
                function_calls.append(group)
            else:
                function_calls.append(ExpectedFunctionCall(**call))

        obj['expected_function_calls'] = function_calls
        return cls(**obj)

    @classmethod
    def create(cls, description, categories, prompt, functions, function_calls):
        return cls(
            description=description,
            categories=set(categories),
            prompt=prompt,
            available_functions=[FunctionDefinition(**func) for func in functions],
            expected_function_calls=[
                ExpectedFunctionCallGroup(any_order=[ExpectedFunctionCall(**fc) for fc in call['any_order']])
                if 'any_order' in call else ExpectedFunctionCall(**call)
                for call in function_calls
            ]
        )
