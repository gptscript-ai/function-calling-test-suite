import os
import pytest
import json
from openai import OpenAI
from benchmark import TestCase


def pytest_addoption(parser):
    parser.addoption("--suite", action="store", default="tests/baseline", help="Directory containing JSON files with test cases")
    parser.addoption("--stream", action="store", default=False, help="Enable streaming for all chat completion requests")

@pytest.fixture(scope="session")
def llm():
    api_key = os.getenv("BENCHMARK_API_KEY")
    base_url = os.getenv("BENCHMARK_BASE_URL")
    return OpenAI(base_url=base_url, api_key=api_key)

@pytest.fixture(scope="session")
def model() -> str | None:
    return os.getenv("BENCHMARK_MODEL")

def pytest_generate_tests(metafunc):
    if "test_case" in metafunc.fixturenames:
        suite = metafunc.config.getoption("--suite")
        stream = bool(metafunc.config.getoption("--stream"))
        test_cases = load_test_cases(suite, stream)
        metafunc.parametrize("test_id, stream, test_case", test_cases, ids=[test_id for test_id, _, _ in test_cases])

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

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if not hasattr(item, 'test_case'):
        return
    
    if report.when != "call":
        return

    test_case = item.test_case
    report.test_case_data = {
       "categories": list(test_case.categories)
    }

    item.user_properties.append(("categories", test_case.categories))

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    passed_tests = terminalreporter.stats.get('passed', [])
    failed_tests = terminalreporter.stats.get('failed', [])

    # Generate the summary data structure from test outcomes
    summary = summarize_results(passed_tests, failed_tests)
    
    # Print the generated summary in the terminal report
    terminalreporter.ensure_newline()
    terminalreporter.section("Benchmark Summary", sep='=', blue=True, bold=True)
    terminalreporter.line(f"total: {summary['passed']}/{summary['total']}")

    for category, counts in sorted(summary["categories"].items()):
        terminalreporter.line(f"{category}: {counts['passed']}/{counts['total']}")
        terminalreporter.ensure_newline()

def summarize_results(passed_tests, failed_tests):
    summary = {
        "total": len(passed_tests) + len(failed_tests),
        "passed": len(passed_tests),
        "failed": len(failed_tests),
        "categories": {}
    }

    # Combine passed and failed tests for processing
    all_tests = passed_tests + failed_tests

    for test in all_tests:
        if hasattr(test, 'test_case_data') and 'categories' in test.test_case_data:
            data = test.test_case_data
            for category in data["categories"]:
                if category not in summary["categories"]:
                    summary["categories"][category] = {"total": 0, "passed": 0, "failed": 0}
                
                # Determine the outcome for incrementing the appropriate counter
                outcome = "passed" if test in passed_tests else "failed"
                summary["categories"][category]["total"] += 1
                summary["categories"][category][outcome] += 1

    return summary

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
