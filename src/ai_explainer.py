import ollama

def call_llm_for_justification(context_data: str, df_metadata: str) -> str:
    '''Calls Llama 3.2 3B to provide a technical justification for a specific decision.'''
    
    # Prompting the model to act as a Senior Consultant
    prompt = (
        'You are a Senior Data Science Consultant for StatGuard. '
        'Based on the provided dataset metadata and a specific pipeline decision, '
        'write a professional 1-sentence justification (under 30 words) for why this '
        'technique was chosen.\n\n'
        f'DATASET CONTEXT:\n{df_metadata}\n\n'
        f'DECISION TO JUSTIFY:\n{context_data}\n\n'
        'Focus on data types, outliers, or distribution. Do not use filler phrases. '
        'Output the justification text only.'
    )

    response = ollama.chat(
        model='llama3.2:3b',
        messages=[{'role': 'user', 'content': prompt}]
    )

    content = response.get('message', {}).get('content', '').strip()
    
    return content if content else 'Justification generation failed.'

def populate_full_justifications(recommendation_skeleton: dict, df_metadata: dict) -> dict:
    '''Iterates through the skeleton dictionary and fills the empty strings using the LLM.'''
    
    metadata_str = str(df_metadata)

    # 1. Justify Global Model Selection
    for strat in recommendation_skeleton.get('global_strategies', []):
        # We justify the top-ranked model option
        top_model = strat['model_options'][0]
        context = f'Global Model Choice: {top_model}'
        strat['justification'] = call_llm_for_justification(context, metadata_str)

    # 2. Justify Column-wise Decisions
    for col_rec in recommendation_skeleton.get('column_strategies', []):
        col_name = col_rec['column_name']
        
        # Extract the top (Rank 1) techniques to justify the reasoning
        top_impute = col_rec['imputation_options'][0]['technique']
        top_scale = col_rec['scaling_options'][0]['technique']
        top_encode = col_rec['encoding_options'][0]['technique']
        
        context = (
            f'Column: {col_name} | '
            f'Imputation: {top_impute} | '
            f'Scaling: {top_scale} | '
            f'Encoding: {top_encode}'
        )
        
        # Fill the 'reasoning' field for the column
        col_rec['reasoning'] = call_llm_for_justification(context, metadata_str)
        
        # Optionally: Fill justifications for individual options if required
        # For now, we focus on the high-level reasoning to keep the UI clean
        col_rec['imputation_options'][0]['justification'] = f'Best performing {top_impute} strategy.'

    return recommendation_skeleton