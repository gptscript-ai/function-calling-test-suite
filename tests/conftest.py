import os
import pytest
import json
from openai import OpenAI
from benchmark import TestCase

def pytest_addoption(parser):
    parser.addoption("--suite", action="store", default="tests/baseline", help="Directory containing JSON files with test cases")

def load_test_cases(suite):
    suite_test_cases = []
    test_case_files = [f for f in os.listdir(suite) if f.endswith('.json')]

    for test_case_file in test_case_files:
        file_path = os.path.join(suite, test_case_file)
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        
        for idx, item in enumerate(json_data):
            try:
                test_case = TestCase.model_validate(item)
                test_id = f"{test_case_file}-{idx}"
                suite_test_cases.append((test_case, test_id))
            except Exception as e:
                print(f"Error parsing {test_case_file} at index {idx}: {e}")

    return suite_test_cases

def pytest_generate_tests(metafunc):
    if "test_case" in metafunc.fixturenames:
        suite = metafunc.config.getoption("--suite")
        test_cases = load_test_cases(suite)
        metafunc.parametrize("test_case, test_id", test_cases, ids=[test_id for _, test_id in test_cases])

@pytest.fixture(scope="session")
def llm():
    api_key = os.getenv("BENCHMARK_API_KEY")
    base_url = os.getenv("BENCHMARK_BASE_URL")
    return OpenAI(base_url=base_url, api_key=api_key)

@pytest.fixture(scope="session")
def model() -> str | None:
    return os.getenv("BENCHMARK_MODEL")
