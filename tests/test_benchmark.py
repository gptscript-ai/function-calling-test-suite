import json
import time
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionMessageToolCall, ChatCompletion, ChatCompletionChunk, ChatCompletionAssistantMessageParam
from benchmark import TestCase, Actual
from collections import deque 

def test_benchmark(llm: OpenAI, model: str | None, test_id: str, stream: bool, test_case: TestCase):
    tools = []
    for function in test_case.available_functions:
        tools.append({
            "type": "function",
            "function": function,
        })

    messages = [{
       "role": "system",
       "content": """
Make the necessary tool calls to execute the available functions in the order specified by the given prompt as correctly and efficiently as possible.
You never explain yourself or provide additional commentary.
You never respond with content.
""".replace("\n", "")
    }, {
       "role": "user",
       "content": test_case.prompt,
    }]

    test_case.actual = Actual(
        tools=tools,
        messages=messages
    )

    call_idx = 1
    expected_calls = deque(test_case.expected_function_calls)
    while expected_calls:
        completion = llm.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0,
            stream=stream
        )

        model_response = to_chat_completion(completion)

        test_case.actual.responses.append(model_response)

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
            assert json.loads(actual_call.function.arguments) == expected_call.arguments, f"Model response tool call should have correct arguments for call {call_idx}"
    
            messages.append({
                "tool_call_id": actual_call.id,
                "role": "tool",
                "name": expected_call.name, 
                "content": expected_call.result
            }) 
            test_case.actual.messages=messages

            call_idx += 1



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

        choices[index]["message"]["tool_calls"] = [tool_calls[index][call_index] for call_index in sorted(tool_calls[index].keys())]
        
    completion = {
        "id": id,
        "created": int(time.time()),
        "object": "chat.completion",
        "model": model,
        "choices": [choices[index] for index in sorted(choices.keys())],
        "system_fingerprint": system_fingerprint
    }

    return ChatCompletion.model_validate(completion)
