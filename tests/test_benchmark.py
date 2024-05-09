import json
import time
from typing import Optional, List
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from benchmark import TestCase, ExpectedFunctionCallGroup, Actual, ActualFunctionCall
from dataclasses import dataclass
from collections import deque


def test_benchmark(
        judge_client: OpenAI,
        model_client: OpenAI,
        model: str | None,
        test_id: str,
        request_delay: float,
        stream: bool,
        use_system_prompt: bool,
        test_case: TestCase
):
    tools = []
    for function in test_case.available_functions:
        tools.append({
            "type": "function",
            "function": function,
        })

    messages = []
    if test_case.system_prompt:
        messages.append({
            "role": "system",
            "content": test_case.system_prompt,
        })
    elif use_system_prompt:
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

    expected_calls = deque(test_case.expected_function_calls or [])
    while True:
        if request_delay > 0.0:
            time.sleep(request_delay)

        model_response = to_chat_completion(model_client.chat.completions.create(
            messages=messages,
            model=model,
            tools=tools,
            tool_choice="auto",
            temperature=0,
            stream=stream,
            n=1,
            timeout=300.0
        ))
        test_case.actual.responses.append(model_response.model_dump(mode='json', exclude_unset=True, exclude_none=True))

        choices = model_response.choices
        assert len(choices) == 1, f"Call {call_index}: Model returned unexpected number of choices"

        choice = choices[0]
        message = choice.message
        assert message.role == "assistant", f"Call {call_index}: Model returned unexpected role: {message.role}"

        messages.append(message.model_dump(mode='json', exclude_unset=True, exclude_none=True))
        if message.content:
            answers.append(message.content)

        tool_calls = deque(message.tool_calls or [])
        for tool_call in tool_calls:
            function_call = tool_call.function
            if not function_call:
                continue

            test_case.actual.function_calls.append(ActualFunctionCall(
                name=function_call.name,
                arguments=json.loads(function_call.arguments)
            ))

        remaining_expected_calls = 0
        for call in expected_calls:
            if hasattr(call, 'any_order'):
                remaining_expected_calls += len(call.any_order)
                continue

            remaining_expected_calls += 1

        assert len(tool_calls) <= remaining_expected_calls, f"Call {call_index}: Model returned more tool calls than expected"

        if len(expected_calls) == 0 or len(tool_calls) == 0:
            assert choice.finish_reason == "stop", f"Call {call_index}: Model returned unexpected finish reason"
            break

        while tool_calls and expected_calls:
            tool_call = tool_calls.popleft()
            next_expected_call = expected_calls.popleft()
            expected_call = None

            if hasattr(next_expected_call, 'any_order'):
                for index, call in enumerate(next_expected_call.any_order):
                    if call.name == tool_call.function.name \
                            and json.loads(tool_call.function.arguments) == call.arguments:
                        expected_call = next_expected_call.any_order.pop(index)
                        break

                assert expected_call is not None, f"Call {call_index}: Tool call not found in expected call group"
                if len(next_expected_call.any_order) > 0:
                    expected_calls.appendleft(next_expected_call)
            else:
                expected_call = next_expected_call
                # Skip optional calls
                while expected_calls:
                    if not hasattr(expected_call, 'any_order') and expected_call.optional \
                            and expected_call.name != tool_call.function.name \
                            and json.loads(tool_call.function.arguments) != expected_call.arguments:
                        expected_call = expected_calls.popleft()
                        call_index += 1
                    break

            assert tool_call.id != "", f"Call {call_index}: Model returned a tool call without a call id"
            assert tool_call.function.name == expected_call.name, f"Call {call_index}: Model returned a tool call with an unexpected function name: {tool_call.function.name}"
            assert json.loads(tool_call.function.arguments) == expected_call.arguments, f"Call {call_index}: Model returned a tool call with unexpected arguments"

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": expected_call.name,
                "content": expected_call.result
            })
            test_case.actual.messages = messages
            call_index += 1

        assert len(tool_calls) == 0, f"Call {call_index}: Model returned unexpected tool calls"

    remaining_required_calls = 0
    for call in expected_calls:
        if hasattr(call, 'any_order'):
            remaining_expected_calls += len([c for c in call.any_order if not c.optional])
        elif not call.optional:
            remaining_expected_calls += 1

    assert remaining_required_calls == 0, f"Model did not make all required tool calls before stopping"

    if test_case.final_answer_should:
        final_answer = '\n'.join(answers)
        correct, reasoning = judge_final_answer(judge_client, stream, final_answer, test_case.final_answer_should)
        test_case.actual.answers = answers
        test_case.actual.judge_ruling = reasoning

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
    "name": "ruling",
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
    judge_ruling = json.loads(judge_message)

    try:
        return judge_ruling['correct'], judge_ruling['reasoning']
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

        for choice in chunk.choices or []:
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
                        "type": "function",
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

    return ChatCompletion(
        id=id,
        created=int(time.time()),
        object="chat.completion",
        model=model,
        choices=[choices[index] for index in sorted(choices.keys())] if choices is not None else [],
        system_fingerprint=system_fingerprint
    )
