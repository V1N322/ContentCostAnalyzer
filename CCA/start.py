import googleAPI
import openAI
import AnthropicAI

from dotenv import load_dotenv

import json
import os
import logging
from datetime import datetime
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

# Configure logging
logging.basicConfig(filename='test_log.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def test_openai_model(model_name, prompt, input_token_cost, output_token_cost):
    """Tests an OpenAI model and returns the result as a JSON string."""
    print(f"DEBUG: test_openai_model called with model: {model_name}, prompt: {prompt[:20]}..., input_cost: {input_token_cost}, output_cost: {output_token_cost}")
    api_key = os.getenv("OPENAI_TOKEN")
    client = openAI.GPTHandler(model_name, api_key)
    try:
        result = client.generate_response(prompt, 8192, input_token_cost, output_token_cost)
        logging.info(f"Successfully tested OpenAI model: {model_name}")
        json_result = client.to_json(result) # Исправлено: вызываем метод to_json у объекта client
        print(f"DEBUG: OpenAI Result JSON: {json_result}")
        return json_result
    except Exception as e:
        logging.error(f"Error testing OpenAI model: {model_name}. Error: {e}")
        print(f"DEBUG: Error testing OpenAI model: {model_name}. Error: {e}")
        return None

def test_anthropic_model(model_name, prompt, input_token_cost, output_token_cost):
    """Tests an Anthropic model and returns the result as a JSON string."""
    print(f"DEBUG: test_anthropic_model called with model: {model_name}, prompt: {prompt[:20]}..., input_cost: {input_token_cost}, output_cost: {output_token_cost}")
    api_key = os.getenv("ANTHROPIC_TOKEN")
    client = AnthropicAI.ClaudeHandler(model_name, api_key)
    try:
        result = client.generate_response(prompt, 8192, input_token_cost, output_token_cost)
        logging.info(f"Successfully tested Anthropic model: {model_name}")
        json_result = client.to_json(result) # Исправлено: вызываем метод to_json у объекта client
        print(f"DEBUG: Anthropic Result JSON: {json_result}")
        return json_result
    except Exception as e:
        logging.error(f"Error testing Anthropic model: {model_name}. Error: {e}")
        print(f"DEBUG: Error testing Anthropic model: {model_name}. Error: {e}")
        return None

def test_google_model(model_name, prompt, input_token_cost, output_token_cost):
    """Tests a Google model and returns the result as a JSON string."""
    print(f"DEBUG: test_google_model called with model: {model_name}, prompt: {prompt[:20]}..., input_cost: {input_token_cost}, output_cost: {output_token_cost}")
    api_key = os.getenv("GOOGLE_TOKEN_PATH")
    client = googleAPI.GoogleAIHandler(model_name, api_key)
    try:
        result = client.generate_response(prompt, input_token_cost, output_token_cost)
        logging.info(f"Successfully tested Google model: {model_name}")
        json_result = client.to_json(result) # Исправлено: вызываем метод to_json у объекта client
        print(f"DEBUG: Google Result JSON: {json_result}")
        return json_result
    except Exception as e:
        logging.error(f"Error testing Google model: {model_name}. Error: {e}")
        print(f"DEBUG: Error testing Google model: {model_name}. Error: {e}")
        return None

def test_models(prompt, models_info):
    """Tests a list of models from a provider and returns a list of results."""
    results = []
    print(f"DEBUG: test_models called with prompt: {prompt[:20]}..., models_info: {models_info}")
    for model in models_info['models']:
        logging.info(f"Testing model: {model['name']} from provider: {models_info['provider']}")
        print(f"DEBUG: Provider: {models_info['provider']}, Model: {model['name']}")
        if models_info['provider'] == "Google":
            result_json = test_google_model(model['name'], prompt, model['pricing']['input']['1000t'], model['pricing']['output']['1000t'])
        elif models_info['provider'] == "Anthropic":
            result_json = test_anthropic_model(model['name'], prompt, model['pricing']['input']['1000t'], model['pricing']['output']['1000t'])
        elif models_info['provider'] == "OpenAI":
            result_json = test_openai_model(model['name'], prompt, model['pricing']['input']['1000t'], model['pricing']['output']['1000t'])
        else:
            logging.warning(f"Unknown provider: {models_info['provider']}")
            print(f"DEBUG: Unknown provider: {models_info['provider']}")
            continue

        print(f"DEBUG: Result JSON before appending: {result_json}")
        if result_json:
            try:
                results.append(json.loads(result_json))
                print(f"DEBUG: Result JSON appended successfully.")
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON: {e}. JSON string: {result_json}")
                print(f"DEBUG: Error decoding JSON: {e}. JSON string: {result_json}")
        else:
            print(f"DEBUG: result_json is None, skipping.")
    return results

def process_prompt_file(prompt_path, thematics_dir, models_dir, all_results, progress_bar):
    """Processes a single prompt file against all thematics and models."""
    logging.info(f"Processing prompt file: {prompt_path}")
    print(f"DEBUG: process_prompt_file called with prompt_path: {prompt_path}, thematics_dir: {thematics_dir}, models_dir: {models_dir}")
    with open(prompt_path, 'r') as p:
        base_prompt = p.read().strip()  # Читаем базовый промпт
        print(f"DEBUG: Base prompt read: {base_prompt[:50]}...")
        for thematic_file in get_files_in_dir(thematics_dir):
            logging.info(f"Processing thematic: {thematic_file}")
            print(f"DEBUG: Processing thematic: {thematic_file}")
            thematic_path = os.path.join(thematics_dir, thematic_file)
            with open(thematic_path, 'r') as t:
                thematic = t.read().strip() # Читаем тематику
            
            # Формируем полный промпт
            prompt = f"{base_prompt}\n{thematic}"
            print(f"DEBUG: Full Prompt: {prompt[:50]}...")

            for model_file in get_files_in_dir(models_dir):
                model_path = os.path.join(models_dir, model_file)
                print(f"DEBUG: Processing model file: {model_path}")
                with open(model_path, 'r') as m:
                    model_info = json.load(m)
                    print(f"DEBUG: Model info loaded: {model_info}")
                    all_results.extend(test_models(prompt, model_info))
                    progress_bar.update(1)
                    print(f"DEBUG: all_results after extend: {all_results}")

def get_files_in_dir(directory):
    """Returns a list of files in the specified directory."""
    print(f"DEBUG: get_files_in_dir called with directory: {directory}")
    files = [f for f in listdir(directory) if isfile(join(directory, f))]
    print(f"DEBUG: Files found: {files}")
    return files

def get_result():
    """Main function to run tests and save results to a file."""
    cur_dir = os.path.join(os.getcwd(), "CCA")
    data_dir = os.path.join(cur_dir, "data")
    prompts_dir = os.path.join(data_dir, 'prompts')
    thematics_dir = os.path.join(data_dir, 'thematics')
    models_dir = os.path.join(data_dir, 'models')
    result_dir = os.path.join(cur_dir, 'result')
    os.makedirs(result_dir, exist_ok=True)

    result_file_name = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    result_file_path = os.path.join(result_dir, result_file_name)

    all_results = []
    print(f"DEBUG: get_result: prompts_dir: {prompts_dir}, thematics_dir: {thematics_dir}, models_dir: {models_dir}")

    # Исправлен расчет итераций:
    total_iterations = sum(len(get_files_in_dir(thematics_dir)) * len(model_info['models']) for _ in get_files_in_dir(prompts_dir) for model_file in get_files_in_dir(models_dir) for model_info in [json.load(open(os.path.join(models_dir, model_file), 'r'))])

    with tqdm(total=total_iterations, desc="Testing Progress") as progress_bar:
        for prompt_file in get_files_in_dir(prompts_dir):
            prompt_path = os.path.join(prompts_dir, prompt_file)
            process_prompt_file(prompt_path, thematics_dir, models_dir, all_results, progress_bar)

    with open(result_file_path, 'w') as result_file:
        json.dump(all_results, result_file, indent=4)
    logging.info(f"Results saved to: {result_file_path}")
    print(f"Results saved to: {result_file_path}")

def main():
    load_dotenv()
    get_result()

if __name__ == "__main__":
    main()