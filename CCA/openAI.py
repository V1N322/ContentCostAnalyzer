from openai import OpenAI
import os
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv

class GPTHandler:
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_TOKEN")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is missing")

    def generate_response(
        self, 
        prompt: str, 
        max_tokens: int = 1000, 
        cost_per_1000_input_tokens: float = 0.01,
        cost_per_1000_output_tokens: float = 0.03
    ) -> Dict[str, Any]:
        try:
            client = OpenAI(api_key=self.api_key)
            
            request_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
            }
            
            # Адаптация под разные модели
            if self.model_name == "o1-mini" or self.model_name == "o1-preview":
                request_params["max_completion_tokens"] = max_tokens
            else:
                request_params["max_tokens"] = max_tokens
            
            response = client.chat.completions.create(**request_params)

            output = response.choices[0].message.content.strip()
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

            input_cost = self._calculate_cost(input_tokens, cost_per_1000_input_tokens)
            output_cost = self._calculate_cost(output_tokens, cost_per_1000_output_tokens)
            total_cost = input_cost + output_cost

            return {
                "model": self.model_name,
                "input": prompt,
                "output": output,
                "inputTokens": input_tokens,
                "outputTokens": output_tokens,
                "totalTokens": total_tokens,
                "inputCost": input_cost,
                "outputCost": output_cost,
                "totalCost": total_cost
            }
        except Exception as e:
            return {"model": self.model_name, "error": str(e)} 

    def _calculate_cost(self, tokens: int, cost_per_1000_tokens: float) -> float:
        return (tokens / 1000) * cost_per_1000_tokens

    @staticmethod
    def to_json(data: Dict[str, Any], metadata=None, indent: int = 4) -> str:
        if metadata:
            data['metadata'] = metadata
        return json.dumps(data, ensure_ascii=False, indent=indent)

def main():
    load_dotenv()

    try:
        handler = GPTHandler(model_name="o1")
        
        prompt = input("Enter a prompt: ")
        
        response = handler.generate_response(prompt)
        print(GPTHandler.to_json(response))

    except Exception as e:
        print(f"Critical error: {e}")

if __name__ == "__main__":
    main()