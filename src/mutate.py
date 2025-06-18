from pydantic_ai import Agent
from dotenv import load_dotenv
import logfire
import os
import random
load_dotenv()

logfire.configure(
    token=os.getenv("LOGFIRE_TOKEN")
)
logfire.instrument_pydantic_ai()

agent = Agent(
    model="gpt-4.1-mini",
    output_type=str,
    model_settings={"temperature": 1, "max_tokens": 1500},
    retries=3
)


def generate(prompt: str, model:str = "gpt-4.1-mini", reasoning:bool = False) -> str:
    if reasoning or "o4" in model:
        output = agent.run_sync(prompt, model=model, model_settings={"max_tokens": 10000}).output
    else:
        output = agent.run_sync(prompt, model=model).output

    if "```python" in output:
        output = output.split("```python")[1].split("```")[0]
    elif "```" in output:
        output = output.split("```")[1].split("```")[0]

    
    return output


