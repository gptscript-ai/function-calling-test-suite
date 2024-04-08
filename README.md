# benchmark

# Running Tests 

Set environment variables to target model

```sh
export BENCHMARK_API_KEY='...'
export BENCHMARK_BASE_URL='https://api.openai.com/v1'
export BENCHMARK_MODEL='gpt-3.5-turbo-0125'
```

Run benchmark suite with streaming in verbose mode and write test report to file

```sh
poetry run pytest -vv -s --junitxml="benchmark-report-${BENCHMARK_MODEL}.xml" --suite="tests/baseline" --stream=true
```