import ollama

def call_llm_for_justification(context_data: str, df_metadata: str) -> str:
    '''Calls Llama 3.2 3B to provide a targeted, 1-sentence justification.'''
    
    prompt = (
        'You are a Senior Data Science Consultant for StatGuard. '
        'Based on the provided dataset metadata, write a single, professional sentence '
        '(under 25 words) explaining WHY this specific decision was made for these exact columns.\n\n'
        f'DATASET CONTEXT:\n{df_metadata}\n\n'
        f'DECISION TO JUSTIFY:\n{context_data}\n\n'
        'Focus strictly on data types, missing values, outliers, or model mechanics. '
        'Output the plain justification text only. No intro, no filler.'
    )

    try:
        response = ollama.chat(
            model='llama3.2:3b',
            messages=[{'role': 'user', 'content': prompt}]
        )
        content = response.get('message', {}).get('content', '').strip()
        return content if content else 'Justification generation failed.'
    except Exception as e:
        return f'Error connecting to local LLM: {str(e)}'


def populate_full_justifications(recommendation_skeleton: dict, df_metadata: dict) -> dict:
    '''Iterates through the pipeline options and generates highly granular justifications.'''
    
    metadata_str = str(df_metadata)

    for pipeline in recommendation_skeleton.get('pipeline_options', []):
        
        # 1. Justify the Overall Model
        model_name = pipeline['model']
        model_ctx = f"Algorithm Selected: {model_name}"
        pipeline['justification'] = call_llm_for_justification(model_ctx, metadata_str)

        # 2. Iterate through Transformations and justify each step
        for trans in pipeline.get('transformations', []):
            feat_type = trans['feature_type']

            # Justify Imputation
            for imp in trans.get('imputation', []):
                strat = imp['strategy']
                if strat != 'none': # Save compute by not explaining 'none'
                    cols = ', '.join(imp['columns'])
                    ctx = f"{feat_type.title()} Imputation | Apply '{strat}' to columns: [{cols}]"
                    imp['justification'] = call_llm_for_justification(ctx, metadata_str)
                else:
                    imp['justification'] = 'Columns contain no missing values; imputation bypassed.'

            # Justify Scaling
            for scl in trans.get('scaling', []):
                strat = scl['strategy']
                if strat != 'none':
                    cols = ', '.join(scl['columns'])
                    ctx = f"{feat_type.title()} Scaling | Apply '{strat}' to columns: [{cols}]"
                    scl['justification'] = call_llm_for_justification(ctx, metadata_str)
                else:
                    scl['justification'] = 'Scaling not required for these features.'

            # Justify Encoding
            for enc in trans.get('encoding', []):
                strat = enc['strategy']
                if strat != 'none':
                    cols = ', '.join(enc['columns'])
                    ctx = f"{feat_type.title()} Encoding | Apply '{strat}' to columns: [{cols}]"
                    enc['justification'] = call_llm_for_justification(ctx, metadata_str)
                else:
                    enc['justification'] = 'Encoding not required for these features.'

    return recommendation_skeleton