import ollama
from typing import List, Mapping, Any
from pydantic import BaseModel

class recommendation_options(BaseModel):
    '''Provides reccomendations for datascience techniques'''
    name: str
    action: str
    reasoning: str

class column_recomendation(BaseModel):
    '''Stores columns related recommendations'''
    column_name: str
    top_three_options: List[recommendation_options]

class global_recomendation(BaseModel):
    '''Stores recommendations related to the entire dataset'''
    category: str
    top_three_options: List[recommendation_options]

class full_pipeline_recommendation(BaseModel):
    '''Stores both global and columwise recommendations'''
    # global recommendations
    global_strategies : List[global_recomendation]

    # column recommendations
    column_strategies : List[column_recomendation]

def call_lm_local(metadata_json: str, extra_feedback: str = None) -> str:
    '''Call Ollama model to generate suggestions for data pipeline'''
    # LLM to review metadata (Used for first run)
    user_content = f'Review this metadata {metadata_json}'

    # Provide feedback of invalid errors (subsiquent runs)
    if extra_feedback:
        user_content += (
            f"""\nVALIDATION FEEDBACK
            \tThe previous plan had the following errors: {extra_feedback}, please regenerate the plan and fix these issues."""
        )

    # Model setup and call
    response : Mapping[str, Any] = ollama.chat (
        model= "qwen2.5-coder:3b",
        format=full_pipeline_recommendation.model_json_schema(), 
        messages=[
            {
                # Give model context and set to provide appropriate respones
                'role' : 'system',
                'content': (
                    "You are a Senior Data Science Consultant. Your goal is to provide a PRECISION data pipeline. "
                    "DO NOT give generic advice. You must examine the 'missing_values', 'has_outliers', "
                    "and 'is_categorical' fields for EVERY column."
                    
                    "\n\nSTRICT LOGIC RULES:"
                    "\n1. IF 'missing_values' == 0: You are FORBIDDEN from suggesting Imputation for that column."
                    "\n2. IF 'has_outliers' == false: You are FORBIDDEN from suggesting Outlier Removal for that column."
                    "\n3. IF 'is_binary' == true: Suggest only 1 strategy (e.g., Label Encoding)."
                    "\n4. If a column is already clean (0 missing, no outliers, correctly typed), OMIT it from the JSON."
                    "\n5. Provide up to 3 options ONLY if the data complexity justifies it."
                    
                    "\n\nOutput ONLY raw JSON matching the schema."
                )
            },
            {
                # Provide data to the model
                'role' : 'user',
                'content' : user_content
            }
        ],
        # turn up percision for iterative validation
        options= {
            'temperature' : 0.2 if extra_feedback else 0.5
        }
    )

    # pull content from LLM reply
    content = response.get('message', {}).get('content', 'Error generating reply from LLM')

    return str(content)