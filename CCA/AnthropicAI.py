from anthropic import Anthropic
import os
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv

class ClaudeHandler:
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("ANTHROPIC_TOKEN")
        
        if not self.api_key:
            raise ValueError("Anthropic API key is missing")

    def generate_response(
        self, 
        prompt: str, 
        max_tokens: int = 1000, 
        cost_per_1000_input_tokens: float = 0.02,
        cost_per_1000_output_tokens: float = 0.05
    ) -> Dict[str, Any]:
        try:
            client = Anthropic(api_key=self.api_key)
            
            response = client.messages.create(
                max_tokens=max_tokens,
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            )

            output = response.content[0].text.strip() if response.content else ""
            
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = input_tokens + output_tokens

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
        handler = ClaudeHandler(model_name="claude-3-5-sonnet-latest")
        
        prompt = input("Введите запрос: ")
        
        response = handler.generate_response(prompt)
        print(ClaudeHandler.to_json(response))

    except Exception as e:
        print(f"Критическая ошибка: {e}")


if __name__ == "__main__":
    main()
