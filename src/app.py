import streamlit as st
import pandas as pd
import json
from typing import Callable
from read_config import read_config
from data_scan import scan_df
from pipeline_builder import optimal_pipeline
from ai_explainer import populate_full_justifications

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

    return df, target_col
 
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

def build_pipeline_optimisation(df: pd.DataFrame, meta_data: dict, target_col: str, config: dict) -> dict:
    '''Runs bayesian optimisations and finds ideal pipeline (also calls LLM to generate justifications)'''

    # Create a new season state to hold the resulting output of optimisation
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'metadata_df' not in st.session_state:
        st.session_state.metadata_df = None

    # Target accuracy slider
    st.subheader("Set Performance Threshold")
    st.write("Optimser will search for pipeline that meets the threshold (A higher threshold may take longer)")

    # build slider and seperate if regression or classification
    if meta_data.get('model_task_type', '') == 'regression':
        target_accuracy = st.slider(
            "Minimum Target R²",
            min_value= 50,
            max_value= 85,
            value= 70,
            step= 1,
            help="The optimisation engine will execute trials until this score is achieved or the maximum trial limit is reached."
            )
    else:
        target_accuracy = st.slider(
            "Minimum Target Accuracy",
            min_value= 50,
            max_value= 85,
            value= 70,
            step= 1,
            help="The optimisation engine will execute trials until this score is achieved or the maximum trial limit is reached."
            )

    # Create slider to set the max amount of trials to prevent infinant looping 
    max_trials = st.slider(
        'Maximum search attempts (Prevent infinant search)',
        min_value= 10,
        max_value= 1000,
        value= 500,
        step= 10,
        help= 'The absolute limit of searches possible for building optimal pipeline within accuracy requirement'
    )

    # Build button to begin optimisation
    if st.button("Generate Optimised Pipeline"):
        with st.status('Building optimised machine learning pipeline', expanded= True) as status:
            try:
                # Write out message to display optimisation target
                st.write(f'Executing pipeline automisation search for a score of {target_accuracy:.2f}')

                # Run optimiser through imported function from "pipeline_builder.py"
                recommendation_raw, metadata_df = optimal_pipeline(
                    df= df,
                    target_col= target_col,
                    meta_data= meta_data,
                    config= config,
                    target_accuracy = target_accuracy,
                    max_trials = max_trials
                )

                # Now we call the LLM to write in justifications for the optimisation recommendations
                final_recommendations = populate_full_justifications(recommendation_skeleton= recommendation_raw, df_metadata= meta_data)

                # Save the recommendations and update system state
                st.session_state.recommendations = final_recommendations
                st.session_state.metadata_df = metadata_df
            
                # Safely extract the score from the very first (highest ranked) pipeline
                top_score = final_recommendations.get('pipeline_options', [{}])[0].get('score', 'Unknown')

                # Update the status box using the extracted score
                status.update(label=f"Successfully found the top pipelines with a highest score of {top_score}", state='complete')

            except Exception as e:
                status.update(label="Optimisation Failed", state='error')
                st.error(f"Encountered error: {e} during optimisation process")

    return st.session_state.recommendations

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

def display_results_timeline(recommendations: dict) -> None:
    '''Displays results of the optimisation in sorted pipelines'''

    # Do not run if no recommendations have been generated
    if not recommendations:
        return

    # Create the header for this stremlitt section
    st.divider()
    st.header('Top Three Pipelines')
    st.write(f'Target variable: **{recommendations.get('target_column', 'Error')}** | Task type: **{recommendations.get('task_type', 'Error')}**')

    # Pull out pipeline options from the recommendations
    pipeline_options = recommendations.get('pipeline_options', [])

    # Return if no valid pipelines found
    if not pipeline_options:
        st.warning('Error: No valid pipelines found')
        return

    # Create dynamic tabs
    tab_titles = [f'Rank {pipeline['rank']} (Score: {pipeline['score']})' for pipeline in pipeline_options]
    tabs = st.tabs(tab_titles)

    # Create timelines for pipelines iteravly
    for tab, pipeline in zip(tabs, pipeline_options):
        with tab:

            # Headers for pipelines
            st.subheader(f'Algorythm selected: {pipeline['model']}')
            st.info(f"**Consultant Reasoning:** {pipeline['justification']}")

            st.markdown("---")
            st.markdown("### Recommended features pipeline")
            
            # Iterate through Numeric and Categorical transformations
            for trans in pipeline.get('transformations', []):
                feat_type = trans.get('feature_type', '').title()
                
                # Only display the section if there are actually columns of this type
                has_imputation = any(imp['strategy'] != 'none' for imp in trans.get('imputation', []))
                has_scaling = any(scl['strategy'] != 'none' for scl in trans.get('scaling', []))
                has_encoding = any(enc['strategy'] != 'none' for enc in trans.get('encoding', []))
                
                # skip if no transformation methods suggested
                if not (has_imputation or has_scaling or has_encoding):
                    continue
                    
                st.markdown(f"#### {feat_type} Variables")
                
                # Imputation Timeline
                for imp in trans.get('imputation', []):
                    if imp['strategy'] != 'none':
                        st.markdown(f"**Step 1: Missing Value Imputation** (`{imp['strategy']}`)")
                        st.caption(f"Applied to features: [{', '.join(imp['columns'])}]")
                        st.success(imp.get('justification', 'Mathematical transformation applied.'))
                        
                # Scaling Timeline
                for scl in trans.get('scaling', []):
                    if scl['strategy'] != 'none':
                        st.markdown(f"**Step 2: Feature Scaling** (`{scl['strategy']}`)")
                        st.caption(f"Applied to features: [{', '.join(scl['columns'])}]")
                        st.success(scl.get('justification', 'Mathematical transformation applied.'))
                        
                # Encoding Timeline
                for enc in trans.get('encoding', []):
                    if enc['strategy'] != 'none':
                        st.markdown(f"**Step 3: Categorical Encoding** (`{enc['strategy']}`)")
                        st.caption(f"Applied to features: [{', '.join(enc['columns'])}]")
                        st.success(enc.get('justification', 'Mathematical transformation applied.'))
                
                st.write("")
    return None

def main() -> None:  
    # Build initial sets basic things for streamlit UI
    build_initial()

    # Build CSV uploader to take user data (also defines target col and amount of validator runs)
    df, target_col = build_csv_uploader()

    # Read in config file
    config = read_config(r'config.yaml')

    if df is not None:
        # Build an overview of the uploaded CSV data
        meta_data = build_dataset_overview(df, target_col)

        # Build the ultimate pipeline and recommendations
        reccomendation_data = build_pipeline_optimisation(df, meta_data, target_col, config)

        # Display results in a timeline
        display_results_timeline(reccomendation_data)

    else:
        st.info('Please upload a .CSV file to begin')

    # Build reset button (if user wants to use new .CSV)
    build_reset_button()    

    return None

if __name__ == "__main__":
    main()