import json
import time
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from benchmark import TestCase, Actual
from collections import deque


def test_benchmark(judge_client: OpenAI, model_client: OpenAI, model: str | None, test_id: str, request_delay: float,
                   stream: bool, use_system_prompt: bool, test_case: TestCase):
    tools = []
    for function in test_case.available_functions:
        tools.append({
            "type": "function",
            "function": function,
        })

    messages = []
    if use_system_prompt:
        messages.append({
            "role": "system",
            "content": """
Make the necessary tool calls to execute the available functions in the order specified by the given prompt as correctly and efficiently as possible.
You never explain yourself or provide additional commentary.
You never respond with content.
""".replace("\n", "")
        })

    messages.append({
        "role": "user",
        "content": test_case.prompt,
    })

    test_case.actual = Actual(
        tools=tools,
        messages=messages
    )

    call_index = 0
    answers = []
    expected_calls = deque(test_case.expected_function_calls)
    while expected_calls:
        if request_delay > 0.0:
            time.sleep(request_delay)

        model_response = to_chat_completion(model_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0,
            stream=stream
        ))
        test_case.actual.responses.append(model_response)

        choices = model_response.choices
        assert len(choices) > 0, f"Call {call_index}: Model returned no choices"

        choice = choices[0]
        content = choice.message.content
        if content:
            answers.append(choice.message.content)

        model_message = choice.message
        messages.append(model_message)
        assert model_message.role == "assistant", f"Call {call_index}: Model returned unexpected role"

        expected_call = expected_calls[0]
        if choice.finish_reason == "stop":
            # This means the model has stopped requesting tool calls.
            # Remove any optional calls from the expected deque
            while expected_calls:
                if not expected_call.optional:
                    break
                expected_call = expected_calls.popleft()

        actual_calls = model_message.tool_calls
        if expected_call.finish_reason == "stop":
            assert len(actual_calls or []) == 0, f"Call {call_index}: Model returned tool calls when it should have stopped"
            break

        assert choice.finish_reason == expected_call.finish_reason, f"Call {call_index}: Model returned unexpected finish_reason"

        for actual_call in actual_calls:
            expected_call = expected_calls.popleft()

            assert expected_call.finish_reason != "stop", f"Call {call_index}: Model returned tool call when expected to stop"
            assert actual_call.function.name == expected_call.name, f"Call {call_index}: Model returned tool call unexpected function name"
            assert actual_call.id != "", f"Call {call_index}: Model returned a tool call without a call ID"
            assert json.loads(actual_call.function.arguments) == expected_call.arguments, f"Call {call_index}: Model returned a tool call with unexpected arguments"

            messages.append({
                "tool_call_id": actual_call.id,
                "role": "tool",
                "name": expected_call.name,
                "content": expected_call.result
            })
            test_case.actual.messages = messages
            call_index += 1

    if test_case.final_answer_should:
        final_answer = '\n'.join(answers)
        correct, reasoning = judge_final_answer(judge_client, stream, final_answer, test_case.final_answer_should)
        assert correct, f"Model's final answer ruled incorrect by judge: \"{reasoning}\""


def judge_final_answer(judge_client: OpenAI, stream: bool, final_answer: str, final_answer_should: str) -> (bool, str):
    if not final_answer_should:
        return None, True

    judge_response = judge_client.chat.completions.create(
        model='gpt-4-turbo-preview',
        response_format={
            "type": "json_object",
        },
        messages=[{
            "role": "system",
            "content": """When given JSON objects that conform to the following JSONSchema:
{
    "name": "judge",
    "type": "object",
    "properties": {
        "final_answer": {
            "type": "string",
            "description": "An answer to judge for correctness."
        },
        "final_answer_should": {
            "type": "string",
            "description": "The constraints that final_answer must completely satisfy to be considered correct."
        }
    },
    "required": [
        "final_answer",
        "final_answer_should"
    ]
}

Determine if `final_answer` completely satisfies the constraints described by `final_answer_should`.
`final_answer` is considered correct if and only if it completely satisfies the constraints described by `final_answer_should`.

After making a determination, respond with a JSON object that conforms to the following JSONSchema:

{
    "name": "judgment",
    "type": "object",
    "properties": {
        "correct": {
            "type": "boolean",
            "description": "Set to true if and only if the answer is considered correct."
        },
        "reasoning": {
            "type": "string",
            "description": "A brief summary of the reasoning used to come to the determination."
        }
    },
    "required": [
        "correct",
        "reasoning"
    ]
}

Your responses are concise and include only the json object described above.
"""
        }, {
            "role": "user",
            "content": json.dumps({
                "final_answer": final_answer,
                "final_answer_should": final_answer_should,
            })
        }],
        stream=stream

    )

    judge_completion = to_chat_completion(judge_response)
    judge_message = judge_completion.choices[0].message.content
    judgment = json.loads(judge_message)

    try:
        return judgment['correct'], judgment['reasoning']
    except KeyError as e:
        raise ValueError(f"Failed to judge final answer. Judge response missing key: {e}")


def to_chat_completion(response: ChatCompletion | Stream[ChatCompletionChunk]) -> ChatCompletion:
    if isinstance(response, ChatCompletion):
        return response

    id = ""
    finish_reason = ""
    model = ""
    system_fingerprint = ""
    choices = {}
    tool_calls = {}

    for chunk in response:
        if chunk is None:
            continue

        id = id or chunk.id
        model = model or chunk.model
        system_fingerprint = system_fingerprint or chunk.system_fingerprint

        for choice in chunk.choices:
            if choice is None or choice.delta is None:
                continue

            if choice.finish_reason is not None and finish_reason != "":
                finish_reason = choice.finish_reason

            if choice.index not in choices:
                choices[choice.index] = {
                    "index": choice.index,
                    "message": {
                        "role": choice.delta.role or "",
                        "content": choice.delta.content or "",
                    },
                    "finish_reason": choice.finish_reason
                }
            else:
                choices[choice.index]["message"]["content"] += choice.delta.content or ""
                choices[choice.index]["finish_reason"] = choice.finish_reason

            if choice.delta.tool_calls is None:
                continue

            for response_call in choice.delta.tool_calls:
                if choice.index not in tool_calls:
                    tool_calls[choice.index] = {}

                arguments = ""
                name = ""
                if response_call.function is not None:
                    arguments = response_call.function.arguments or ""
                    name = response_call.function.name or ""

                if response_call.index not in tool_calls[choice.index]:

                    tool_calls[choice.index][response_call.index] = {
                        "id": response_call.id,
                        "type": response_call.type,
                        "function": {
                            "arguments": arguments,
                            "name": name,
                        }
                    }
                else:
                    tool_calls[choice.index][response_call.index]["function"]["arguments"] += arguments

    for index, choice in choices.items():
        if index not in tool_calls:
            continue

        if "message" not in choices[index]:
            continue

        choices[index]["message"]["tool_calls"] = [tool_calls[index][call_index] for call_index in
                                                   sorted(tool_calls[index].keys())]

    completion = {
        "id": id,
        "created": int(time.time()),
        "object": "chat.completion",
        "model": model,
        "choices": [choices[index] for index in sorted(choices.keys())],
        "system_fingerprint": system_fingerprint
    }

    return ChatCompletion.model_validate(completion)
