from typing_extensions import TypedDict


class State(TypedDict):
    user_input: str | None
    classification: str | None
    response: str | None
