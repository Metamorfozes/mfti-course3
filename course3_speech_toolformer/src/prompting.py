SYSTEM_PROMPT = (
    "Output must be either exactly NO_TOOL or a single JSON object with keys "
    "name and arguments. JSON format: "
    '{"name":"unit_convert","arguments":{"value":<float>,"from_unit":"<unit>","to_unit":"<unit>","precision":2}} '
    "No extra text before/after JSON. If no numeric conversion requested -> NO_TOOL."
)


def build_messages(user_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
