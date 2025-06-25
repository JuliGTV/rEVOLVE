from pydantic_ai import Agent
from dotenv import load_dotenv
import logfire
import os
import asyncio
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


async def generate_async(prompt: str, model: str = "gpt-4.1-mini", reasoning: bool = False) -> str:
    """Async version of generate function with better model handling"""
    import time
    start_time = time.time()
    
    try:
        if reasoning or "o4" in model:
            logfire.debug(f"Starting reasoning model call: {model}")
            output = (await agent.run(prompt, model=model, model_settings={"max_tokens": 30000})).output
        else:
            logfire.debug(f"Starting standard model call: {model}")
            output = (await agent.run(prompt, model=model)).output

        duration = time.time() - start_time
        logfire.debug(f"Model call completed in {duration:.2f}s: {model}")

        if "```python" in output:
            output = output.split("```python")[1].split("```")[0]
        elif "```" in output:
            output = output.split("```")[1].split("```")[0]

        return output
    except Exception as e:
        duration = time.time() - start_time
        logfire.error(f"Model call failed after {duration:.2f}s: {model} - {str(e)}")
        raise
