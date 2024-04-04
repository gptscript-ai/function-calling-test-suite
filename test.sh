#!/bin/bash

set -e
set -o pipefail

# Capture the PID of the run script to stop gptscript provider daemons later.
script_pid=$$

# Cleanup provider daemons to allow this script to exit after tests run.
cleanup() {
  echo "Cleaning up provider daemons..."
  pkill -P ${script_pid}
  echo "Cleanup complete!"
}

run_suite() {
  local provider="$1"
  local model="$2"

  gptscript -q --disable-tui --disable-cache --daemon "${provider}" | {
    trap cleanup EXIT
    read LINE
    export FCTS_BASE_URL="${LINE}/v1" FCTS_MODEL="${model}" FCTS_API_KEY='foo'
    poetry run pytest --cache-clear -s --stream=true "${@:3}" && echo "${model} FCTS test complete"
    exit
  }
}

setup_gemini() {
  if ! gcloud config list 2>/dev/null | grep -qE 'quota_project|account|project'; then
    cat <<EOF
Invalid gcloud config.

Please set the billing/quota_project, core/account, and core/project fields before attempting to run FCTS against VertexAI.

e.g.

$ gcloud config set project <project-name>
$ gcloud config set billing/quota_project <project-name>
$ gcloud config set account <account-email>

$ gcloud config list

[billing]
quota_project = acorn-io
[core]
account = nick@acorn.io
disable_usage_reporting = False
project = acorn-io

Your active configuration is: [default]
EOF
    exit 1
  fi

  if ! gcloud auth application-default print-access-token >/dev/null 2>&1; then
    gcloud auth application-default login
  fi
}

case "$1" in
clear) clear_env ;;
mistral-large-latest) run_suite github.com/gptscript-ai/mistral-laplateforme-provider mistral-large-2402 "${@:2}" ;;
gemini-1.5-pro)
  setup_gemini
  run_suite github.com/gptscript-ai/gemini-vertexai-provider gemini-1.5-pro "${@:2}"
  ;;
claude-3.5-sonnet) run_suite github.com/gptscript-ai/claude3-anthropic-provider claude-3-5-sonnet-20240620 "${@:2}" ;;
*)
  echo "Usage: $0 {clear|mistral-large-latest|gemini-1.5-pro|claude-3.5-sonnet} [PYTEST_OPTIONS]"
  exit 1
  ;;
esac
