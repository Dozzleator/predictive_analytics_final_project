import streamlit as st 

def render_css_timeline(pipeline: dict) -> None:
    '''Create a neet CSS timeline for the pipeline steps'''

    # html string (will be ingected into streamlit) to build the pipeline
    html_content = """<style>
.timeline-container {
    font-family: sans-serif;
    margin: 20px 0 20px 10px;
    padding-left: 25px;
    border-left: 3px solid #FF4B4B; /* Changed to Red */
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
    background-color: #FF4B4B; /* Changed to Red */
    border: 3px solid #0E1117;
}
.timeline-content {
    background-color: #000000; /* Changed to pure Black */
    border: 1px solid #333333; /* Added a subtle dark grey border to make the box pop */
    padding: 15px;
    border-radius: 8px;
}
.step-title {
    margin: 0 0 5px 0 !important;
    font-size: 16px;
    font-weight: 600;
    color: #FF4B4B; /* Changed to Red */
}
.step-caption {
    margin: 0 0 10px 0 !important;
    font-size: 13px;
    opacity: 0.8;
}
.step-justification {
    margin: 0 !important;
    font-size: 14px;
    font-style: italic;
    border-left: 2px solid rgba(255, 75, 75, 0.3); /* Changed to a subtle red accent line */
    padding-left: 10px;
}
</style>
<div class="timeline-container">
"""

    # track the steps to update dynamically where required
    step_counter = 1

    # Inject if pipeline had logrithmic transformation (non-linear)
    if pipeline.get('distribution_transformed'):
        target_just = pipeline.get('Distribution_transformation_justification', 'Distribution transformation applied.')

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

                    # Append data to html
                    html_content += f"""<div class="timeline-item">
    <div class="timeline-dot"></div>
    <div class="timeline-content">
        <h4 class="step-title">Step {step_counter}: {display_name} ({strategy_name})</h4>
        <p class="step-caption">Applied to {feat_type} features: [{cols}]</p>
        <p class="step-justification">"{justification}"</p>
    </div>
</div>
"""

                    step_counter += 1

    # Build in model justification to the timeline
    model_name = pipeline.get('model', 'Model')
    model_justification = pipeline.get('model_selection_justification', 'Algorithm selected.')
    
    # Cap the timeline
    html_content += f"""<div class="timeline-item">
    <div class="timeline-dot"></div>
    <div class="timeline-content">
        <h4 class="step-title">Step {step_counter}: Train {model_name}</h4>
        <p class="step-caption">Final mathematical model trained on the engineered features above.</p>
        <p class="step-justification">"{model_justification}"</p>
    </div>
</div>
</div>
"""

    # Allow for html injection
    st.markdown(html_content, unsafe_allow_html=True)