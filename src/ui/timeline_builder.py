import streamlit as st 

def render_css_timeline(pipeline: dict) -> None:
    '''Create a neet CSS timeline for the pipeline steps'''

    # html string (will be ingected into streamlit) to build the pipeline
    html_content = """<style>
.timeline-container {
    font-family: sans-serif;
    margin: 20px 0 20px 10px;
    padding-left: 25px;
    border-left: 3px solid #FF4B4B; 
}
.timeline-item {
    position: relative;
    margin-bottom: 25px;
}
.timeline-dot {
    position: absolute;
    left: -35px;
    top: 5px;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background-color: #FF4B4B; 
    border: 3px solid #0E1117;
}
.timeline-content {
    background-color: #000000; 
    border: 1px solid #333333; 
    padding: 15px;
    border-radius: 8px;
}
.step-title {
    margin: 0 0 5px 0 !important;
    font-size: 16px;
    font-weight: 600;
    color: #FF4B4B; 
}
.step-caption {
    margin: 0 0 10px 0 !important;
    font-size: 13px;
    opacity: 0.8;
}
.step-justification {
    margin: 0 !important;
    font-size: 14px;
    border-left: 2px solid rgba(255, 75, 75, 0.3); 
    padding-left: 10px;
}
</style>
<div class="timeline-container">
"""

    # track the steps to update dynamically where required
    step_counter = 1

    # Inject if pipeline had logrithmic transformation (non-linear)
    if pipeline.get('distribution_transformed'):
        target_just = pipeline.get('distribution_transformation_justification', 'Distribution transformation applied.')

        # Pull the specific math function for the title (Log1p or Arcsinh)
        transform_type = pipeline.get('transformation_used', 'Log1p').title()

        html_content += f"""<div class="timeline-item">
<div class="timeline-dot"></div>
<div class="timeline-content">
<h4 class="step-title">Step {step_counter}: Target Transformation ({transform_type})</h4>
<p class="step-caption">Applied to the Target Variable</p>
<p class="step-justification">"{target_just}"</p>
</div>
</div>
"""
        step_counter += 1

    # Loop through transformations to dynamically build required steps
    for t in pipeline.get('transformations', []):
        feat_type = t.get('feature_type', '').title()

        # Iterate through all keys in the transformation dictionary
        for step_type, items in t.items():

            # Skip feature_type identifier as it is not a proccessing step 
            if step_type == 'feature_type':
                continue
        
            # Format the key into a clean title
            display_name = step_type.replace('_', ' ').title()

            # now loop through the strategies and justifications 
            for item in items:
                if item['strategy'] != 'none':
                    strategy_name = item['strategy'].replace('_', ' ').title()
                    cols = ', '.join(item['columns'])
                    justification = item.get('justification', 'Error generating a justification for selected strategy')

                    step_params = item.get('hyperparameters', {})
                    param_html = ''
                    if step_params:
                        param_html += '<div style="margin-top: 15px; font-size: 13px; color: #A0A0A0; border-top: 1px dashed #333333; padding-top: 10px;">'
                        param_html += '<strong>Configuration:</strong><br>'
                        for p_name, p_val in step_params.items():
                            if p_val is not None:
                                clean_val = f'{p_val:.4f}' if isinstance(p_val, float) else str(p_val)
                                param_html += f'&#8226; {p_name.replace("_", " ").title()}: {clean_val}<br>'
                        param_html += '</div>'

                    # Append data to html
                    html_content += f"""<div class="timeline-item">
<div class="timeline-dot"></div>
<div class="timeline-content">
<h4 class="step-title">Step {step_counter}: {display_name} ({strategy_name})</h4>
<p class="step-caption">Applied to {feat_type} features: [{cols}]</p>
<p class="step-justification">"{justification}"</p>
{param_html}
</div>
</div>
"""
                    step_counter += 1

# Build Class Balancing into the frontend timeline
    balance_strat = pipeline.get('class_balancing_strategy', 'None')
    balance_just = pipeline.get('class_balancing_justification', '')

    # Only draw this HTML block if a balancing strategy was actually applied
    if 'None' not in balance_strat:
        html_content += f"""<div class="timeline-item">
<div class="timeline-dot"></div>
<div class="timeline-content">
<h4 class="step-title">Step {step_counter}: Class Balancing</h4>
<p class="step-caption">Applied Strategy: {balance_strat}</p>
<p class="step-justification">"{balance_just}"</p>
</div>
</div>
"""
        step_counter += 1

    # Build in model justification to the timeline
    model_name = pipeline.get('model', 'Model')
    model_justification = pipeline.get('model_selection_justification', 'Algorithm selected.')
    model_params = pipeline.get('model_hyperparameters', {})
        
    # Dynamically build the hyperparameter HTML block if they exist
    model_param_html = ''
    if model_params:
        model_param_html += '<div style="margin-top: 15px; font-size: 13px; color: #A0A0A0; border-top: 1px dashed #333333; padding-top: 10px;">'
        model_param_html += '<strong>Optimised Hyperparameters:</strong><br>'
        for param, value in model_params.items():
            if value is not None:
                clean_value = f'{value:.4f}' if isinstance(value, float) else str(value)
                model_param_html += f'&#8226; {param}: {clean_value}<br>'
        model_param_html += '</div>'
    
    # Add in model step with injected hyperparameters
    html_content += f"""<div class="timeline-item">
<div class="timeline-dot"></div>
<div class="timeline-content">
<h4 class="step-title">Step {step_counter}: {model_name} Model</h4>
<p class="step-caption">Final mathematical model trained on the engineered features above.</p>
<p class="step-justification">"{model_justification}"</p>
{model_param_html}
</div>
</div>
"""
    step_counter += 1

    # Pull validation strategies
    val_strat = pipeline.get('validation_strategy', 'Standard Cross Validation')
    val_just = pipeline.get('validation_justification', '')

    val_params = pipeline.get('validation_hyperparameters', {})
    val_param_html = ''
    if val_params:
        val_param_html += '<div style="margin-top: 15px; font-size: 13px; color: #A0A0A0; border-top: 1px dashed #333333; padding-top: 10px;">'
        val_param_html += '<strong>Validation Configuration:</strong><br>'
        for param, value in val_params.items():
            if value is not None:
                clean_param = param.replace('_', ' ').title()
                val_param_html += f'&#8226; {clean_param}: {value}<br>'
        val_param_html += '</div>'

    # Add in validation step
    html_content += f"""<div class="timeline-item">
<div class="timeline-dot"></div>
<div class="timeline-content">
<h4 class="step-title">Step {step_counter}: Pipeline Validation</h4>
<p class="step-caption">Evaluated using {val_strat}</p>
<p class="step-justification">"{val_just}"</p>
{val_param_html}
</div>
</div>
</div>
"""

    # Allow for html injection
    st.markdown(html_content, unsafe_allow_html=True)