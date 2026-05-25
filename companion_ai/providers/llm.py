import json
import re
from typing import Any, Optional


class GroqChatProvider:
    def __init__(self, api_key: str, model: str):
        self.model = model
        self.client = None
        self.bad_request_error = None

        if not api_key:
            print("GROQ_API_KEY is not set. Using offline text-only replies.")
            return

        try:
            from groq import BadRequestError, Groq
        except ImportError:
            print("Groq package is not installed. Using offline text-only replies.")
            return

        self.bad_request_error = BadRequestError
        try:
            self.client = Groq(api_key=api_key)
        except Exception as exc:
            print(f"Groq client could not be initialized: {exc}")
            print("Using offline text-only replies.")

    def reply(self, messages: list[dict[str, str]]) -> str:
        return self.complete(messages).get("content", "")

    def complete(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        if self.client is None:
            return self._offline_complete(messages, tools)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 500,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        try:
            completion = self.client.chat.completions.create(
                **kwargs,
            )
        except Exception as error:
            if (
                self.bad_request_error is not None
                and isinstance(error, self.bad_request_error)
                and tools
                and "tool_use_failed" in str(error)
            ):
                print("LLM tool call failed; retrying response without tools.")
                kwargs.pop("tools", None)
                kwargs.pop("tool_choice", None)
                kwargs["messages"] = messages + [
                    {
                        "role": "system",
                        "content": (
                            "Your previous tool call could not be processed. Reply in "
                            "normal user-facing text only. Do not include function tags, "
                            "XML, or JSON tool calls in the text."
                        ),
                    }
                ]
                try:
                    completion = self.client.chat.completions.create(
                        **kwargs,
                    )
                except Exception as retry_error:
                    print(f"Groq request failed: {retry_error}")
                    print("Using offline text-only reply for this turn.")
                    return self._offline_complete(messages, tools)
            else:
                print(f"Groq request failed: {error}")
                print("Using offline text-only reply for this turn.")
                return self._offline_complete(messages, tools)
        message = completion.choices[0].message
        return {
            "content": message.content or "",
            "tool_calls": self._tool_calls(message),
        }

    @staticmethod
    def _offline_complete(
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        if tools and any(
            tool.get("function", {}).get("name") == "start_conversation"
            for tool in tools
        ):
            prompt = messages[-1].get("content", "") if messages else ""
            match = re.search(r"Dominant non-neutral emotion label:\s*(.+)", prompt)
            emotion = match.group(1).strip() if match else "different"
            return {
                "content": "",
                "tool_calls": [
                    {
                        "name": "start_conversation",
                        "arguments": {
                            "opener": (
                                "I might be misreading this, but you seem a little "
                                f"{emotion}. Do you want to talk about what's going on?"
                            )
                        },
                    }
                ],
            }

        user_text = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_text = message.get("content", "")
                break

        normalized = user_text.lower()
        if any(word in normalized for word in ("suicide", "self-harm", "kill myself")):
            content = (
                "I'm really sorry you're dealing with this. If you might be in immediate "
                "danger, contact local emergency services or a trusted person now."
            )
        elif any(
            phrase in normalized
            for phrase in ("bye", "goodbye", "no thanks", "thank you", "thanks")
        ):
            content = "Take care of yourself. I'm here if you need anything later."
        else:
            content = (
                "I'm running in offline mode, so I can't use the Groq model right now. "
                "I can still listen here. What's been weighing on you most?"
            )
        return {"content": content, "tool_calls": []}

    @staticmethod
    def _tool_calls(message: Any) -> list[dict[str, Any]]:
        tool_calls = getattr(message, "tool_calls", None) or []
        parsed_calls = []
        for tool_call in tool_calls:
            function = getattr(tool_call, "function", None)
            if function is None and isinstance(tool_call, dict):
                function = tool_call.get("function", {})

            name = getattr(function, "name", None)
            arguments = getattr(function, "arguments", "{}")
            if isinstance(function, dict):
                name = function.get("name")
                arguments = function.get("arguments", "{}")

            if isinstance(arguments, dict):
                parsed_arguments = arguments
            else:
                try:
                    parsed_arguments = json.loads(arguments or "{}")
                except json.JSONDecodeError:
                    parsed_arguments = {}

            if name:
                parsed_calls.append({"name": name, "arguments": parsed_arguments})

        return parsed_calls
