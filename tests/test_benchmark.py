import json
from openai import OpenAI
from benchmark import TestCase
from collections import deque 


def test_benchmark(llm: OpenAI, model: str | None, test_case: TestCase, test_id: str):
    tools = []
    for function in test_case.available_functions:
        tools.append({
            "type": "function",
            "function": function,
        })

    messages = [{
       "role": "user",
       "content": test_case.prompt,
    }] 

    call_idx = 1
    expected_calls = deque(test_case.expected_function_calls)
    print(expected_calls)
    while expected_calls:
        model_response = llm.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        print(model_response.model_dump_json(indent=4))

        choices = model_response.choices
        assert len(choices) > 0, f"Model response should have a choice for call {call_idx}"

        choice = choices[0]
        assert choice.finish_reason == "tool_calls", f"Model response should have correct finish_reason for call {call_idx}"
            
        model_message = choice.message 
        assert model_message.role == "assistant", f"Model response should have correct role for call {call_idx}"
            
        actual_calls = model_message.tool_calls
        assert len(actual_calls) > 0, f"Model response should have at least one tool call for call {call_idx}"

        messages.append(model_message)

        for actual_call in actual_calls:
            expected_call = expected_calls.popleft()

            assert actual_call.id != "", f"Model response tool call should have an id set for call {call_idx}" 
            assert actual_call.type == "function", f"Model response tool call should have correct type for call {call_idx}"
            assert actual_call.function.name == expected_call.name, f"Model response tool should call the expected function for call {call_idx}"

            actual_args = json.loads(actual_call.function.arguments)
            print(actual_args)
            print(expected_call.arguments)

            assert actual_args == expected_call.arguments, f"Model response tool call should have correct arguments for call {call_idx}"

            messages.append({
                "tool_call_id": actual_call.id,
                "role": "tool",
                "name": expected_call.name, 
                "content": expected_call.result
            }) 

            call_idx += 1
