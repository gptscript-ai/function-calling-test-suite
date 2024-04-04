from pydantic import BaseModel
from typing import List, Dict, Any

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]  # Adjust the type as needed

class ExpectedFunctionCall(BaseModel):
    name: str
    arguments: Dict[str, Any] 
    result: str

class TestCase(BaseModel):
    __test__ = False
    prompt: str
    available_functions: List[FunctionDefinition]
    expected_function_calls: List[ExpectedFunctionCall]