# function-benchmark

## Usage 

Set environment variables to target model

```sh
export BENCHMARK_API_KEY='...'
export BENCHMARK_BASE_URL='https://api.openai.com/v1'
export BENCHMARK_MODEL='gpt-4-turbo-2024-04-09'
```

Run `function-benchmark` in verbose mode using for spec files in `./specs` with streaming enabled.

```sh
poetry shell
poetry install
poetry run pytest -vvv -s --stream=true
```

## Testing models without chat completion API support

`gptscript's` [alternative model providers](https://docs.gptscript.ai/alternative-model-providers) can be used to
test models that don't support OpenAI's chat completion API. Providers act as a shim layer between `function-benchmark`
and the model.

## Testing `gemini-1.5` with the `gemini-vertexai-provider`

### Requirements

- [gcloud CLI](https://cloud.google.com/sdk/docs/install-sdk)
- [VertexAI](https://cloud.google.com/vertex-ai) access

### Setup

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

Run the provider server:

```sh
./run.sh
```

### Running the tests

In another terminal, set the `function-benchmark` environment variables to target the provider shim:

```sh
export BENCHMARK_MODEL='gemini-1.5-pro-preview-0409'
export BENCHMARK_BASE_URL='http://127.0.0.1:8081/v1'
export BENCHMARK_API_KEY='foo'
```

 > **Note: The API key can be set to any arbitrary value, but must be set**

Run the tests:

```shell
poetry shell
poetry install
poetry run pytest -s -vvv --stream=true --spec-run-count=10
```

> **Note: The `gemini-vertexai-provider` doesn't support non-streaming responses, so `function-benchmark` streaming must be enabled**
