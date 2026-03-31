import streamlit as st
import pandas as pd
import json
from data_scan import scan_df
from ai_call_local import call_lm_local
# from ai_call_api import call_lm_api as call_lm_local
from validator import result_validator
from typing import Callable

def build_initial() -> None:
    '''Build in setreamlit name for program and quick description'''
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

def build_csv_uploader() -> tuple[pd.DataFrame, str, int, int]:
    '''Build csv uploader and define target column (for predictions)'''
    # Parse uploaded .CSV dataset
    uploaded_file = st.file_uploader('Upload your Dataset (CSV)', type='csv', key = 'csv_uploader')

    # Set to zero so no unbound error befor running 
    df = None
    target_col = None
    run_attempts = 0
    input_attempts = 0

    # Validate setup with input for run options
    if uploaded_file:
        # Read file once for performance
        @st.cache_data
        def load_data(file):
            return pd.read_csv(file)

        # set session state to read df
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
        if target_col:
            if target_col not in df_cols and target_col is not None:
                st.error(f"Could not find input prediction column {target_col} in dataset")
                st.stop()
            else:
                st.success(f"prediction column found: {target_col}")
        else:
            st.info("Please input name of the column you wish to predict to continue")
            st.stop()

        input_attempts = st.slider('Select amount of times for iterative validation attempts (as runs locally more will consume more memory)', 1, 10, 5)
        run_attempts = input_attempts
    return df, target_col, run_attempts, input_attempts

def build_dataset_overview(df: pd.DataFrame, target_col: str) -> json:
    # Create view of metadata pulled from table
    st.subheader("Overview of Metadata")
    meta_table1, meta_table2, meta_table3 = st.tabs(["Column Summary", "Numerical Columns Overview", "String Columns Overview"])
    meta_data = scan_df(df, target_col)

    # Create a table of Dataset statistics
    with meta_table1:

        # Set into own fields
        m1, m2, m3, m4 = st.columns(4)

        # Print out statistics for dataset
        with m1: 
            st.metric(f'Prediction Variable',  f'{meta_data.get('prediction_variable', 'N/A')}')
        with m2: 
            st.metric(f'Model Type', f'{meta_data.get('model_task_type', 'N/A')}')
        with m3:
            st.metric(f'Total Rows', f'{meta_data.get('total_rows', 'N/A')}')
        with m4:
            st.metric(f'Total Columns', f'{meta_data.get('total_columns', 'Unkown')}')

    # Create a table of numerical column statistics
    with meta_table2:
        if 'columns' in meta_data:
            # Put metadata into dataframe
            stats_df = pd.DataFrame(meta_data["columns"])

            # Define to find only integer and float values
            integer_obj_mask = stats_df['type'].str.startswith(('int', 'float'))
            numerical_df = stats_df[integer_obj_mask]

            # Set x-axis as column name (instead of int index)
            numerical_df = numerical_df.set_index('name')

            # Transpose dataframe for readability
            numerical_df = numerical_df.T

            # Highlight column if is target_col 
            numerical_df = numerical_df.style.apply(
                lambda col: ['background-color: rgba(130, 224, 145, 0.3)' if col.name == target_col else '' for _ in col], 
                axis=0
            )

            # Build dataframe of how the data should be displayed
            st.dataframe(
                numerical_df, 
                use_container_width=True,
                height=400,
                column_config={
                    'missing_values' : st.column_config.NumberColumn("Null Values"),
                    'unique_values' : st.column_config.NumberColumn("Unique Values"),
                    'mean' : st.column_config.NumberColumn('Mean of Dataset', format='%.2f')
                }
            )
        else:
            st.warning('No column statistics for numerical columns found in the meta data')

    # Create a table of object(categorical/ str) column statistics
    with meta_table3:
        if 'columns' in meta_data:
            # Put metadata into dataframe
            stats_df = pd.DataFrame(meta_data["columns"])

            # Define to find only integer and float values
            object_obj_mask = stats_df['type'].str.startswith(('obj', 'str'))
            object_df = stats_df[object_obj_mask]

            # Set x-axis as column name (instead of int index)
            object_df = object_df.set_index('name')

            # Transpose dataframe for readability
            object_df = object_df.T

            # Drop NAN rows (as something like mean will only have values for numerical columns)
            object_df = object_df.dropna(how='all')

            # Highlight column if is target_col 
            object_df = object_df.style.apply(
                lambda col: ['background-color: rgba(130, 224, 145, 0.3)' if col.name == target_col else '' for _ in col], 
                axis=0
            )

            # Build dataframe of how the data should be displayed
            st.dataframe(
                object_df, 
                use_container_width=True,
                height=400,
                column_config={
                    'missing_values' : st.column_config.NumberColumn("Null Values"),
                    'unique_values' : st.column_config.NumberColumn("Unique Values"),
                    'mean' : st.column_config.NumberColumn('Mean of Dataset', format='%.2f')
                }
            )
        else:
            st.warning('No column statistics for categorical columns found in the meta data')

    st.divider()
    return meta_data

def build_LLM_validator(df: pd.DataFrame, meta_data: json, target_col: str, LLM_call: Callable, input_attempts: int, run_attempts: int) -> json:
    '''Call API and put recommendations generated into Json'''

    # Initilise with nothing (prevent unbound error)
    recommendations = None

    # Create save for recommendations generated by LLM
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None

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
                st.json(recommendations)
                st.session_state.recommendations = recommendations
            except Exception as e:
                status.update(label="Validation Failed", state= 'error')
                st.error(f"Encountered error as {e} during validation process")
    return st.session_state.recommendations

def build_LLM_result_view(recommendations: json, df: pd.DataFrame, target_col: str) -> None:
    '''Create a view to see the AI recommendations in sorted format'''

    # Exit if not run recommendations yet
    if recommendations is None:
        return

    # Create view for global recomendations
    st.subheader('Global Strategies')
    if 'global_strategies' in recommendations:

        # Create a new tab for each global recommendation
        global_categories = [strategy['category'].title() for strategy in recommendations['global_strategies']]
        global_tabs = st.tabs(global_categories)

        # Create a tab for each strategy and show top 3 recommendations in each tab
        for i, strategy in enumerate(recommendations['global_strategies']):
            with global_tabs[i]:
                cols = st.columns(3)
                for idx, option in enumerate(strategy.get('top_three_options', [])[:3]):
                    with cols[idx]:
                        st.subheader(option['name'])
                        with st.expander("View Strategy Details", expanded=True):
                            st.write(f"**Action:** {option['action']}")
                            st.write(f"**Reasoning:** {option['reasoning']}")
                st.divider()

    # Create view for column wise recommendations 
    st.subheader('Feature-Level Recomendations')
    st.info('Click a column name to see the recomendations generated for the column')

    # Remove prediction column (should not transform only usefull for supervised machine learning)
    view_df = df.drop(columns=target_col)

    # Set up dataframe view
    event = st.dataframe(
        view_df.head(5),
        use_container_width= True,
        on_select= 'rerun',
        selection_mode='single-column',
        hide_index= True
    )

    # Column selection logic
    selected_columns = event.get('selection', {}).get('columns', [])
    if selected_columns:    
        selected_feature = selected_columns[0]
        st.success(f'Veiwing recomendations for: **{selected_feature}**')

        # Find recomendation for selected column
        column_list = recommendations.get('column_strategies', [])
        match = next((item for item in column_list if item['column_name'] == selected_feature), None)

        # Show recommendation if column selection matches recommendation
        if match:
            for option in match.get('top_three_options', []):
                with st.expander(f'{option.get('name')}', expanded= True):
                    st.write(f"**Action:** {option.get('action')}")
                    st.write(f"**Explination:** {option.get('reasoning')}")
    return None

def build_reset_button() -> None:
    '''Reset button to reset seeion state if user wants to upload new .CSV'''
    # Reset program if user wants to upload new CSV data
    st.divider()
    button_space1, button_space2, button_space3 = st.columns([1, 2, 3]) 
    with button_space1:
        if st.button("Scan New CSV", type = "primary", use_container_width= True):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.cache_data.clear()
            st.rerun()

def main() -> None: 
    # Set kind of LLM call (run local or through API)
    LLM_call = call_lm_local

    # Build initial sets basic things for streamlit UI
    build_initial()

    # Build CSV uploader to take user data (also defines target col and amount of validator runs)
    df, target_col, run_attempts, input_attempts = build_csv_uploader()

    if df is not None:
        # Build an overview of the uploaded CSV data
        meta_data = build_dataset_overview(df, target_col)

        # Run validator 
        recommendations = build_LLM_validator(df, meta_data, target_col, LLM_call, input_attempts, run_attempts)

        # Build view for ouput recommendations
        build_LLM_result_view(recommendations, df, target_col)

    else:
        st.info('Please upload a .CSV file to begin')

    # Build reset button (if user wants to use new .CSV)
    build_reset_button()    

    return None

if __name__ == "__main__":
    main()