import torch
from transformers import pipeline


class LLMModel:
    def __init__(self, model_name: str, task_name: str, max_length: int, temperature: float, top_p: float):
        # Initialize the model here (e.g., load weights, set up tokenizer, etc.)
        self.model = pipeline(
            model=model_name,
            task=task_name,
            dtype=torch.bfloat16,
            device=0,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )

    def get_model_response(self, prompt: list) -> str:
        # Initialize the pipeline
        outputs = self.model(prompt)
        return outputs[0]["generated_text"][-1]["content"]
