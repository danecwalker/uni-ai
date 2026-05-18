from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    handler: Callable[..., str]


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def list_descriptions(self) -> str:
        if not self._tools:
            return "No specialised tools are currently available."
        return "\n".join(f"- {tool.name}: {tool.description}" for tool in self._tools.values())

    def call(self, name: str, **kwargs: Any) -> str:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name].handler(**kwargs)


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    # Example placeholder. Add real tools here, e.g. wellbeing plans,
    # reminders, calendar integrations, journaling, university services, etc.
    registry.register(
        Tool(
            name="grounding_prompt",
            description="Offers a short grounding exercise when the user feels overwhelmed.",
            handler=lambda: "Try naming five things you can see, four you can feel, three you can hear, two you can smell, and one you can taste.",
        )
    )
    return registry
