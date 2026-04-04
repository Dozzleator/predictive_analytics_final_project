import json
from read_config import read_config
from data_scan import read_csv, scan_df
from pipeline_builder import optimal_pipeline
from ai_explainer import populate_full_justifications

def main() -> None:
    # Set run parameters
    input_dir = "simple_linear_regression/india_population.csv"
    n_trials = 1000
    target_accuracy = 75
    path_config = r"config.yaml"

    # Starting program
    print("""\n\033[92mStarting Program\033[00m""")

    # load in config
    config = read_config(path_config)

    # Column used for predictions
    target_column = "median_age"

    # Read in CSV data
    df = read_csv(input_dir)
    print(f"\n\033[91mPrinting the first 10 rows of selected CSV:\033[00m")
    print(df.head(10))

    # Scan CSV to create dictionary of meta-data
    df_metadata = scan_df(df, target_column)
    print(f"\n\033[93mPrinting Meta-data that will be sent to LLM:\033[00m")
    print(json.dumps(df_metadata, indent= 4))

    # Run optimser iterativley until the score threshold is higher then 60% 
    print(f'\n\033[94mRunning Optuna Search ({n_trials} trials)...\033[00m')
    recommendations, metadata_df = optimal_pipeline(
        df=df, 
        target_col=target_column, 
        meta_data=df_metadata, 
        config=config,
        target_accuracy= target_accuracy,
        max_trials=n_trials
    )

    # Display Optuna Results
    # Convert dataset metadata dict to string once for the LLM
    print('\n\033[95mPipeline Action Log (Metadata Table):\033[00m')
    print(metadata_df.to_string(index=False))

    print('\n\033[93mGenerating AI Consultant Explanations for Top 3 Pipelines...\033[00m')
    
    # build pipeline and get recomendations from llm
    final_report = populate_full_justifications(recommendations, df_metadata)

    # Output Results
    print('\n\033[92mFinal Justified Report Generated:\033[00m')
    print(json.dumps(final_report, indent=4))

    print('\n\033[96mPipeline Action Log (Metadata Table):\033[00m')
    print(metadata_df.to_string(index=False))

    return None

if __name__ == "__main__":
    main()