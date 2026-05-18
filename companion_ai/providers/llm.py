from groq import Groq


class GroqChatProvider:
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY is required")
        self.client = Groq(api_key=api_key)
        self.model = model

    def reply(self, messages: list[dict[str, str]]) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )
        return completion.choices[0].message.content or ""
