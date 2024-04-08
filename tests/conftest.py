import os
import pytest
import json
from openai import OpenAI
from benchmark import TestCase

def pytest_addoption(parser):
    parser.addoption("--suite", action="store", default="tests/baseline", help="Directory containing JSON files with test cases")
    parser.addoption("--stream", action="store", default=False, help="Enable streaming for all chat completion requests")

def load_test_cases(suite: str, stream: bool):
    suite_test_cases = []
    test_case_files = [f for f in os.listdir(suite) if f.endswith('.json')]

    for test_case_file in test_case_files:
        file_path = os.path.join(suite, test_case_file)
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        
        for idx, item in enumerate(json_data):
            try:
                test_id = f"{test_case_file}-{idx}"
                test_case = TestCase.model_validate(item)
                suite_test_cases.append((test_id, stream, test_case))
            except Exception as e:
                print(f"Error parsing {test_case_file} at index {idx}: {e}")

    return suite_test_cases

# Assuming you have a fixture that receives `test_case` and `test_id` parameters
@pytest.fixture(autouse=True)
def attach_test_case_to_node(request, test_case):
    request.node.test_case = test_case

def pytest_exception_interact(node, call, report):
    if report.failed:
        if hasattr(node, 'test_case'):
           test_case = node.test_case
           messages = [message.model_dump() if hasattr(message, 'model_dump') else message for message in test_case.actual.messages]
           print(json.dumps(messages, indent=4))

           responses = [response.model_dump() for response in test_case.actual.responses]
           print(json.dumps(responses, indent=4))

def pytest_generate_tests(metafunc):
    if "test_case" in metafunc.fixturenames:
        suite = metafunc.config.getoption("--suite")
        stream = bool(metafunc.config.getoption("--stream"))
        test_cases = load_test_cases(suite, stream)
        metafunc.parametrize("test_id, stream, test_case", test_cases, ids=[test_id for test_id, _, _ in test_cases])

@pytest.fixture(scope="session")
def llm():
    api_key = os.getenv("BENCHMARK_API_KEY")
    base_url = os.getenv("BENCHMARK_BASE_URL")
    return OpenAI(base_url=base_url, api_key=api_key)

@pytest.fixture(scope="session")
def model() -> str | None:
    return os.getenv("BENCHMARK_MODEL")
