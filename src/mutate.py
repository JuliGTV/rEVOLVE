from pydantic_ai import Agent
from dotenv import load_dotenv
import logfire
import os
load_dotenv()

logfire.configure(
    token=os.getenv("LOGFIRE_TOKEN")
)
logfire.instrument_pydantic_ai()

model = Agent(
    model="gpt-4o-mini",
    output_type=str
)


def generate(prompt: str, reason: bool = False) -> str:
    output = model.run_sync(prompt).output

    if reason:
        return output.split(r"\n---\n")[1]
    else:
        return output


