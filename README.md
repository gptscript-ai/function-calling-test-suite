# function-calling-test-suite

`function-calling-test-suite` (`FCTS`) is a pragmatic test framework for assessing the function calling capabilities of large language models (LLMs).

## Test spec overview

Test specs files contain YAML streams that define the metadata, input, and expected output for a set of test cases.

e.g.

```yaml
---
categories:
  - basic
description: >-
  Asserts that the model can make a function call with a given argument and
  conveys the result to the user
prompt: Call funcA with 1 and respond with the result of the call
available_functions:
  - name: funcA
    description: Performs funcA
    parameters:
      type: object
      properties:
        param1:
          type: integer
          description: Param 1
expected_function_calls:
  - name: funcA
    arguments:
      param1: 1
    result: This is the output of funcA(1)
final_answer_should: >-
  The answer should indicate that the result of calling funcA with 1 is "This is
  the output of funcA(1)"
---
# ...
```

### Spec anatomy

Every test spec has three primary components:

#### 1. Test metadata

Each test spec must include a `description` and `categories`. The `description` outlines the test's goal, while
`categories` tag the capabilities being tested. Categorizing test cases helps identify the model's strengths and weaknesses.

#### 2. Functions definitions and expected calls
When executed, the framework uses the `prompt` and `available_functions` to generate an initial request.
It compares the model’s response to `expected_function_calls`. If they match, the framework continues making requests
with the `result` field until all expected calls are completed or a call fails.

#### 3. Answer criteria

Even if a model completes all expected function calls, the final response still needs to be verified.
To this end, specs can optionally include a `final_answer_should` field to describe valid answers using natural language.

### Default test suite

The default suite of spec files can be found in the [specs](./specs) directory.

## Basic usage

Initialize test environment

```sh
poetry shell
poetry install
```

Configure FCTS to judge model responses with `gpt-4-turbo`

```sh
export OPENAI_API_KEY='<openai-api-key>'
```

Target a model to test 

```sh
export FCTS_API_KEY='<model-provider-api-key>'
export FCTS_BASE_URL='<model-provider-api-base-url>'
export FCTS_MODEL='<model-name>'
```

Run the [default test suite](./specs) with verbose output enabled:

```sh
poetry run pytest -vvv
```

### Run options

```sh
$ poetry run pytest -h
usage: pytest [options] [file_or_dir] [file_or_dir] [...]
...
Custom options:
  --spec-run-count=SPEC_RUN_COUNT                   Number of times each test spec should be run
  --spec-filter=SPEC_FILTER                         Filter which test specs are run by their generated test IDs
  --spec-dir=SPEC_DIR                               Directory containing JSON test spec files
  --stream=STREAM                                   Enables streaming for all chat completion requests
  --use-system-prompt=USE_SYSTEM_PROMPT             Add a default system prompt to all chat completion requests
  --aggregate-summary-file=AGGREGATE_SUMMARY_FILE   Add results for the model to an aggregate CSV file
  --request-delay=REQUEST_DELAY                     Delay in seconds between chat completion requests
...
```

## Testing models without chat completion API support

GPTScript's [alternative model provider shims](https://docs.gptscript.ai/alternative-model-providers) can be used to test models that don't support OpenAI's chat
completion API.

### claude-3.5-sonnet

Set an Anthopic key:

```shell
export ANTHROPIC_API_KEY='<anthropic-key>'
```

Clone the [claude3-anthropic-provider](https://github.com/gptscript-ai/claude3-anthropic-provider):

```sh
git clone https://github.com/gptscript-ai/claude3-anthropic-provider
```

Follow the `Development` instructions in the repo's `README.md`:

```sh
cd claude3-anthropic-provider
export GPTSCRIPT_DEBUG=true
python -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt
```

Run the shim:

```sh
./run.sh
```

In another terminal, target the provider shim:

```sh
export FCTS_MODEL='claude-3-5-sonnet-20240620'
export FCTS_BASE_URL='http://127.0.0.1:8000/v1'
export FCTS_API_KEY='foo'
```

> **Note:** The API key can be set to any arbitrary value, but must be set

Run the tests:

```shell
poetry shell
poetry install
poetry run pytest --stream=true
```

> **Note: Streaming must be enabled because the `claude3-anthropic-provider` doesn't support non-streaming responses**

### gemini-1.5

Ensure the following requirements are met:

- [gcloud CLI](https://cloud.google.com/sdk/docs/install-sdk)
- [VertexAI](https://cloud.google.com/vertex-ai) access

Configure `gcloud` CLI to use your VertexAI project and account:

```sh
gcloud config set project <project-name> 
gcloud config set billing/quota_project <project-name> 
gcloud config set account <account-email> 
gcloud components update
```

Afterwords, your configuration should look something like this:

```sh
gcloud config list
[billing]
quota_project = acorn-io
[core]
account = nick@acorn.io
disable_usage_reporting = False
project = acorn-io

Your active configuration is: [default]
```

Authenticate with the `gcloud` CLI:

```sh
gcloud auth application-default login
```

Clone the [gemini-vertexai-provider repo](https://github.com/gptscript-ai/gemini-vertexai-provider):

```sh
git clone https://github.com/gptscript-ai/gemini-vertexai-provider
```

Follow the `Development` instructions in the repo's `README.md`:

```sh
cd gemini-vertexai-provider
export GPTSCRIPT_DEBUG=true
python -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt
```

Run the shim:

```sh
./run.sh
```
In another terminal, target the provider shim:

```sh
export FCTS_MODEL='gemini-1.5-pro-preview-0409'
export FCTS_BASE_URL='http://127.0.0.1:8081/v1'
export FCTS_API_KEY='foo'
```

> **Note:** The API key can be set to any arbitrary value, but must be set

Run the tests:

```shell
poetry shell
poetry install
poetry run pytest --stream=true
```

> **Note:** Streaming must be enabled because the `gemini-1.5-pro-preview-0409` doesn't support non-streaming responses
