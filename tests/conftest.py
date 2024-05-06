import os
import pytest
import json
import fnmatch
import csv
import re
from openai import OpenAI
from benchmark import TestCase


def pytest_addoption(parser):
    parser.addoption("--spec-run-count", action="store", default=1, type=int, help="Number of times each test spec should be run")
    parser.addoption("--spec-filter", action="store", default="*", help="Filter which test specs are run by their generated test IDs")
    parser.addoption("--spec-dir", action="store", default="specs", help="Directory containing JSON test spec files")
    parser.addoption("--stream", action="store", default=False, help="Enables streaming for all chat completion requests")
    parser.addoption("--use-system-prompt", action="store", default=False, help="Adds a default system prompt for all chat completion requests")
    parser.addoption("--aggregate-summary-file", action="store", default="reports/aggregate_summary.csv", help="Add benchmark results for the model to an aggregate CSV file")
    parser.addoption("--request-delay", action="store", default=0.0, help="Delay in seconds between chat completion requests")


@pytest.fixture(scope="session")
def model_client():
    api_key = os.getenv("BENCHMARK_API_KEY")
    base_url = os.getenv("BENCHMARK_BASE_URL")
    return OpenAI(base_url=base_url, api_key=api_key)


@pytest.fixture(scope="session")
def judge_client():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    return OpenAI(base_url=base_url, api_key=api_key)


@pytest.fixture(scope="session")
def model() -> str | None:
    return os.getenv("BENCHMARK_MODEL")


# Assuming you have a fixture that receives `test_case` and `test_id` parameters
@pytest.fixture(autouse=True)
def attach_test_case_to_node(request, test_case):
    request.node.test_case = test_case


def pytest_collection_modifyitems(session, config, items):
    items.sort(key=lambda x: x.nodeid)


def pytest_generate_tests(metafunc):
    if "test_case" in metafunc.fixturenames:
        spec_run_count = metafunc.config.getoption("--spec-run-count")
        spec_filter = metafunc.config.getoption("--spec-filter")
        spec_dir = metafunc.config.getoption("--spec-dir")
        request_delay = float(metafunc.config.getoption("--request-delay"))
        stream = bool(metafunc.config.getoption("--stream"))
        use_system_prompt = bool(metafunc.config.getoption("--use-system-prompt"))
        test_cases = load_test_cases(spec_run_count, spec_filter, spec_dir, request_delay, stream, use_system_prompt)
        metafunc.parametrize("test_id, request_delay, stream, use_system_prompt, test_case", test_cases, ids=[test_id for test_id, _, _, _, _ in test_cases])


def load_test_cases(spec_run_count: int, spec_filter: str, spec_dir: str, request_delay: float, stream: bool, use_system_prompt: bool):
    suite_test_cases = []
    test_case_files = [f for f in os.listdir(spec_dir) if f.endswith('.json')]

    for test_case_file in test_case_files:
        file_path = os.path.join(spec_dir, test_case_file)
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        
        for index, item in enumerate(json_data):
            try:
                test_id = f"{test_case_file}-{index}"
                if not fnmatch.fnmatch(test_id, spec_filter):
                    continue

                test_case = TestCase.model_validate(item)
                for run in range(spec_run_count):
                    suite_test_cases.append((f"{test_id}-{run}", request_delay, stream, use_system_prompt, test_case.model_copy(deep=True)))

            except Exception as e:
                print(f"Error parsing {test_case_file} at index {index}: {e}")

    return suite_test_cases


@pytest.hookimpl(optionalhook=True)
def pytest_json_runtest_metadata(item, call):
    if call.when != 'call':
        return {}

    return {
        'model': item.funcargs.get('model', 'N/A'),
        'test_case': item.test_case.model_dump(mode='json'),
        'start': call.start,
        'stop': call.stop
    }


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if not hasattr(item, 'test_case'):
        return
    
    if report.when != "call":
        return

    test_case = item.test_case
    test_id = item.nodeid

    if report.failed:
        if test_case.description:
            report.longrepr.addsection('Test Description', test_case.description)

        report.longrepr.addsection('Prompt', test_case.prompt)

        if test_case.actual.judgment:
            report.longrepr.addsection('Expected Answer', test_case.final_answer_should)
            report.longrepr.addsection('Actual Answer', '. '.join(test_case.actual.answers))
            report.longrepr.addsection('Judge Ruling', test_case.actual.judgment)

        test_case_dict = item.test_case.model_dump(mode='json')

        expected_calls = test_case_dict.get('expected_function_calls', [])
        available_functions = test_case_dict.get('available_functions', [])
        actual = test_case_dict.get('actual', {})
        messages = actual.get('messages', [])
        actual_calls = actual.get('function_calls', [])
        responses = actual.get('responses', [])

        report.longrepr.addsection('Expected Calls', json.dumps(expected_calls, indent=4))
        report.longrepr.addsection('Actual Calls', json.dumps(actual_calls, indent=4))
        report.longrepr.addsection('Available Functions', json.dumps(available_functions, indent=4))
        report.longrepr.addsection('Request Messages', json.dumps(messages, indent=4))
        report.longrepr.addsection('Raw Responses', json.dumps(responses, indent=4))

    # Pre-process results for aggregate CSV report
    model = item.funcargs.get('model', 'N/A')
    result = "PASSED" if report.outcome == "passed" else "FAILED"
    run_result = {
        "test_id": test_id,
        "categories": " ".join(test_case.categories),
        "description": getattr(test_case, 'description', 'No description'),
        "prompt": getattr(test_case, 'prompt', 'No prompt'),
        "result": result,
        "model": model,
    }

    if not hasattr(item.session.config, 'run_results'):
        item.session.config.run_results = {}
    item.session.config.run_results[(item.nodeid, model)] = run_result


def pytest_sessionfinish(session, exitstatus):
    csv_path = session.config.getoption("--aggregate-summary-file")
    aggregate_results = {}
    columns = ['test_id', 'categories', 'description', 'prompt']  # Default fields
    processed = set()  # Set to track processed test_id, model combinations

    # Check if the CSV exists and read existing data
    if os.path.exists(csv_path):
        with open(csv_path, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            columns = reader.fieldnames  # Capture all existing columns
            for row in reader:
                aggregate_results[row['test_id']] = row

    # Update test_results with new data from the current session
    json_report_path = None
    if hasattr(session.config, 'run_results'):
        for (nodeid, model), run_result in session.config.run_results.items():
            if not json_report_path:
                json_report_path = f"reports/{model}_benchmark_report.json"

            if model not in columns:
                columns.append(model)  # Add new model to fieldnames if it's not already there

            # Extract the simplified test_id from nodeid
            match = re.search(r'\[(.*?)-\d+\]', nodeid)
            simplified_test_id = match.group(1) if match else nodeid
            key = (simplified_test_id, model)

            if simplified_test_id not in aggregate_results:
                aggregate_results[simplified_test_id] = {c: "" for c in columns}
                aggregate_results[simplified_test_id]['test_id'] = simplified_test_id
                aggregate_results[simplified_test_id]['categories'] = run_result['categories']
                aggregate_results[simplified_test_id]['description'] = run_result['description']
                aggregate_results[simplified_test_id]['prompt'] = run_result['prompt']

            # Check if this is the first run of this combination in the current session
            if key not in processed:
                # Reset the counts for the first run of this combination
                passed, total_runs = 0, 0
                processed.add(key)
            else:
                # Get current counts if not the first run
                passed, total_runs = map(int, aggregate_results[simplified_test_id][model].split('/') if aggregate_results[simplified_test_id][model] else (0, 0))

            # Increment runs and update pass rate for test spec
            total_runs += 1
            if run_result['result'] == 'PASSED':
                passed += 1

            # Update the fraction in the CSV
            aggregate_results[simplified_test_id][model] = f"{passed}/{total_runs}"

    # Write the model's benchmark report
    if json_report_path:
        plugin = session.config._json_report
        plugin.save_report(json_report_path)

    # Write updated data back to CSV
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for test_id in sorted(aggregate_results.keys()):
            writer.writerow(aggregate_results[test_id])
