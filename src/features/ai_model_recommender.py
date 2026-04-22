import json
import ollama

def get_dynamic_model_recommendations(config:dict, meta_data: dict, is_classification: bool) -> list:
    '''Call LLM to dynamically select the best models to test with the config file'''

    # Define the task type for the model to look through
    task = 'classificiation' if is_classification else 'regression'

    # Find available models
    models_dict = config.get('models', {})
    if task == 'classificiation':
        models = models_dict.get('classification', [])
    else:
        models = models_dict.get('regression', [])

    # Create the prompt that will be sent to the ollama model
    prompt = (
            f'You are a Senior Data Science Consultant. Review this dataset metadata:\n{meta_data}\n\n'
            f'This is a {task} task. Select the top 3 most appropriate algorithms to test '
            f'from this exact list: {models}.\n\n'
            'Output ONLY a valid JSON format list of the string names. No explanation.'
        )

    try:
        # Return logical results from the model
        response = ollama.chat(
            model='llama3.2:3b',
            messages=[{'role':'user', 'content': prompt}]
        )

        # Parse the results from model
        content = response.get('message', {}).get('content', '').strip()
        recommended_models = json.loads(content)

        # Validate the model selections exist and is from the config
        valid_models = [m for m in recommended_models if m in models]
        if valid_models:
            return valid_models
        else:
            return models
    except Exception:
        models