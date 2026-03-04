import streamlit as st
import pandas as pd
import json
from data_scan import scan_df
from ai_call_local import call_lm_local
# from ai_call_api import call_lm_api
from validator import result_validator

# Set kind of LLM call (run local or through API)
LLM_call = call_lm_local

# Set configuration for streamlit UI
st.set_page_config(page_title= "Machine Learning Recommender System", layout= 'wide')

# Initilise session state 
if 'df' not in st.session_state:
    st.session_state.df = None
if 'meta_data' not in st.session_state:
    st.session_state.meta_data = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None

# Set page title and a short description
st.title("Machine Learning Recommender System")
st.markdown("Automated validation and recommendation system for machine learning pipelines based on uploaded .CSV data")

# Parse uploaded .CSV dataset
uploaded_file = st.file_uploader('Upload your Dataset (CSV)', type='csv', key = 'csv_uploader')

# Validate setup with input for run options
if uploaded_file:
    # Read file once for performance
    @st.cache_data
    def load_data(file):
        return pd.read_csv(file)

    # set session stat to read df
    if st.session_state.df is None:
        st.session_state.df = load_data(uploaded_file)

    # Read data and find columns
    df = st.session_state.df
    df_cols = df.columns.to_list()

    # Preview table once loaded into UI
    with st.expander("Uploaded .CSV dataset", expanded= True):
        st.dataframe(df, use_container_width=True, height= 300)

    # Run settings section of UI
    st.header("Run Input Settings")
    target_col = st.text_input('Input name of prediction column (Case sensitive)')

    # Validate that input target column exists
    if target_col not in df_cols:
        st.error(f"Could not find input prediction column {target_col} in dataset")
        st.stop()
    else:
        st.success(f"prediction column found: {target_col}")

    input_attempts = st.slider('Select amount of times for iterative validation attempts (as runs locally more will consumer more memory)', 1, 20, 10)
    run_attempts = input_attempts

    # Create view of metadata pulled from table
    st.subheader("Overview of Metadata")
    meta_table1, meta_table2 = st.tabs(["Column Summary", "Technical Data"])
    meta_data = scan_df(df, target_col)

    # Create a table of Dataset statistics
    with meta_table1:
        col1, col2 = st.columns(2)
        with col1: 
            st.write('Dataset Information')
            st.write(f'Total Rows in Dataset: {meta_data.get('total_rows', 'N/A')}')
            st.write(f'Total Columns in Dataset: {meta_data.get('total_columns', 'N/A')}')
        with col2:
            st.write(f'Selected Prediction Variable: {meta_data.get('target_col', 'N/A')}')

    # Create a table of column statistics
    with meta_table2:
        if 'columns' in meta_data:
            # Transpose so columns are readable
            stats_df = pd.DataFrame(meta_data["columns"]).T

            # Build dataframe of how the data should be displayed
            st.dataframe(
                stats_df, 
                use_container_width=True,
                height=400,
                column_config={
                    'missing_values' : st.column_config.NumberColumn("Null Values"),
                    'unique_values' : st.column_config.NumberColumn("Unique Values"),
                    'mean' : st.column_config.NumberColumn('Mean of Dataset', format='%.2f')
                }
            )
        else:
            st.warning('No column statistics found in the meta data')

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

    # Reset program if user wants to upload new CSV data
    st.divider()
    button_space1, button_space2, button_space3 = st.columns([1, 2, 3]) 
    with button_space1:
        if st.button("Scan New CSV", type = "primary", use_container_width= True):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.cache_data.clear()
            st.rerun()

