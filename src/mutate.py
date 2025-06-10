from pydantic_ai import Agent



model = Agent(
    model="gpt-4o-mini",
    output_type=str
)


def generate(prompt: str, reason: bool = False) -> str:
    output = model.generate(prompt)

    if reason:
        return output.split(r"\n---\n")[1]
    else:
        return output


