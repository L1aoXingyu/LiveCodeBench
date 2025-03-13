import os
from time import sleep

try:
    import openai
    from openai import OpenAI
except ImportError as e:
    pass

from lcb_runner.runner.base_runner import BaseRunner

class SiliconFlowRunner(BaseRunner):
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url="https://api.siliconflow.cn/v1",
    )

    def __init__(self, args, model):
        super().__init__(args, model)
        self.client_kwargs: dict[str | str] = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            "timeout": args.openai_timeout,
            "stream": True,
            # "stop": args.stop, --> stop is only used for base models currently
        }

    def _run_single(self, prompt: list[dict[str, str]]) -> list[str]:
        if isinstance(prompt, list):
            pass
        else:
            prompt = [{"role": "user", "content": prompt}]

        def __run_single(counter):
            if counter <= 0:
                raise RuntimeError("Maximum retry attempts exceeded")

            try:
                response = self.client.chat.completions.create(
                    messages=prompt,
                    **self.client_kwargs,
                )
                reasoning_content = ""
                content = ""

                for chunk in response:
                    # Extract the delta from choices[0]
                    delta = chunk.choices[0].delta

                    # If there is a reasoning_content field, add it to the accumulator.
                    # (It is assumed that either "reasoning_content" or "content" will be provided per chunk.)
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        reasoning_content += delta.reasoning_content
                    # Otherwise, if there's a content field, add it both to the accumulator and to the file.
                    elif hasattr(delta, "content") and delta.content:
                        text = delta.content
                        content += text
                return content
            except (
                openai.APIError,
                openai.RateLimitError,
                openai.InternalServerError,
                openai.OpenAIError,
                openai.APIStatusError,
                openai.APITimeoutError,
                openai.InternalServerError,
                openai.APIConnectionError,
            ) as e:
                print("Exception: ", repr(e))
                print("Sleeping for 30 seconds...")
                print("Consider reducing the number of parallel processes.")
                sleep(30)
                return __run_single(counter - 1)
            except Exception as e:
                print(f"Failed to run the model for {prompt}!")
                print("Exception: ", repr(e))
                raise e

        outputs = []
        try:
            for _ in range(self.args.n):
                outputs.append(__run_single(10))
        except Exception as e:
            raise e
        return outputs
