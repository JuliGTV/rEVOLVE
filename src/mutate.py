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

model = Agent(
    model="gpt-4.1",
    output_type=str,
    model_settings={"temperature": 1, "max_tokens": 1500}
)


def generate(prompt: str, reason: bool = False, second_model:str = "openai:o4-mini-2025-04-16", second_model_rate:float = 0.15) -> str:
    if random.random() < second_model_rate:
        output = model.run_sync(prompt, model=second_model, model_settings={"max_tokens": 10000}).output
    else:
        output = model.run_sync(prompt).output
    if "```python" in output:
        output = output.split("```python")[1].split("```")[0]
    elif "```" in output:
        output = output.split("```")[1].split("```")[0]
    # if reason:
    #     return output.split(r"\n---\n")[1]
    
    return output


