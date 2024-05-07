# benchmark

# Running Tests 

Set environment variables to target model

```sh
export BENCHMARK_API_KEY='...'
export BENCHMARK_BASE_URL='https://api.openai.com/v1'
export BENCHMARK_MODEL='gpt-4-turbo-2024-04-09'
```

Run benchmark tests in verbose mode using for spec files in `./specs` with streaming enabled.

```sh
poetry run pytest -vv -s --stream=true
```