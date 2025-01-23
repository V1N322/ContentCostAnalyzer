import google.auth
import google.generativeai as genai
import os
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv

class GoogleAIHandler:
    def __init__(self, model_name: str, credentials_file: str):
        self.model_name = model_name
        self.credentials_file = credentials_file
        self.model = None
        self._load_credentials()
        self._configure_model()

    def _load_credentials(self):
        try:
            self.credentials, self.project = google.auth.load_credentials_from_file(self.credentials_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Ошибка: Файл не найден: {self.credentials_file}")
        except Exception as e:
            raise Exception(f"Ошибка при загрузке учетных данных: {e}")

    def _configure_model(self):
        genai.configure(credentials=self.credentials)
        self.model = genai.GenerativeModel(self.model_name)

    def generate_response(
        self,
        prompt: str,
        cost_per_1000_input_tokens: float = 0.00125,
        cost_per_1000_output_tokens: float = 0.005,
    ) -> Dict[str, Any]:
        if not self.model:
            return {"error": "Модель не сконфигурирована."}

        try:
            response = self.model.generate_content(prompt)

            output = response.text.strip() if response and hasattr(response, 'text') else ""

            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            
            if response and hasattr(response, 'candidates') and response.candidates:
                output = response.candidates[0].content.parts[0].text.strip()

            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
                total_tokens = response.usage_metadata.total_token_count

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
    credentials_path = os.getenv("GOOGLE_TOKEN_PATH")
    if not credentials_path:
        print("Переменная окружения GOOGLE_TOKEN_PATH не установлена.")
        return

    try:
        handler = GoogleAIHandler(model_name="gemini-1.5-pro", credentials_file=credentials_path)

        prompt = input("Введите запрос: ")

        response_data = handler.generate_response(prompt)
        print(GoogleAIHandler.to_json(response_data))

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Критическая ошибка: {e}")

if __name__ == "__main__":
    main()