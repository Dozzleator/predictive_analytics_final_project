import streamlit as st
import pandas as pd
import json
from data_scan import scan_df
from ai_call_local import call_lm_local
from ai_call_api import call_lm_api
from validator import result_validator

# Set kind of LLM call (run local or through API)
LLM_call = call_lm_local

# Set configuration for streamlit UI
st.set_page_config(page_title= "Machine Learning Recommender System", layout= 'wide')

# Set page title and a short description
st.title("Machine Learning Recommender System")
st.markdown("Automated validation and recommendation system for machine learning pipelines based on uploaded .CSV data")

# Parse uploaded .CSV dataset
uploaded_file = st.file_uploader('Upload your Dataset (CSV)', type='csv')

# Validate setup with input for run options
if uploaded_file:
    # Read file once for performance
    @st.cache_data
    def load_data(file):
        return pd.read_csv(file)

    # Read data and find columns
    df = load_data(uploaded_file)
    df_cols = df.columns.to_list()

    st.header("Settings")
    target_col = st.text_input('Input name of prediction column (Case sensitive)')

    # Validate that input target column exists
    if target_col not in df_cols:
        st.error(f"Could not find input prediction column {target_col} in dataset")
        st.stop()
    else:
        st.success(f"prediction column found: {target_col}")

    input_attempts = st.slider('Select amount of times for iterative validation attempts (as runs locally more will consumer more memory)', 1, 20, 10)
    run_attempts = input_attempts

    # define a previe of data and the associated metadata
    data_view, metadata_view = st.columns([1, 2])

    # Create view of data in dataset 
    with data_view:
        st.subheader("Overview of uploaded dataset")
        st.dataframe(df.head(10).tail(10) , use_container_width= True)

    # Create view of metadata pulled from table
    with metadata_view:
        st.subheader("Overview of Metadata")
        meta_data = scan_df(df, target_col)
        st.json(meta_data)

    st.divider()

# Loop to excecute the validator
if st.button("Generate validated pipeline"):
    with st.status("Analysing data and generating recommendations please wait", expanded= True) as status:
        validator = result_validator(df, {}, target_col, meta_data)

        try: 
            # Runs validator itteratively to the amount of max attempts
            recommendations, attempts = validator.run_validator(LLM_call, run_attempts)
            if input_attempts < run_attempts:
                status.update(label= f"Sucessfully validated in {run_attempts} attempts")
            else:
                status.update(label=f"Could not validate in {input_attempts} attempts, returning results of final run")
            st.subheader("Final pipeline of recommendations")
            st.json(recommendations)
        except Exception as e:
            status.update(label="Validation Failed", state= 'error')
            st.error(f"Encountered error as {e} during validation process")

else:
    st.info("Please upload CSV file to begin")