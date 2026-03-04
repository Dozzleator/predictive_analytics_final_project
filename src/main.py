import json
import time
from ai_call_api import call_lm_api
from ai_call_local import call_lm_api
from data_scan import read_csv, scan_df
from validator import result_validator

def main() -> None:
    # Set run parameters
    max_ai_calls = 50
    model_function = call_lm_api
    input_dir = "student_dropout_data/student_dropout_dataset.csv"

    # Starting program
    print("""\n\033[92mStarting Program\033[00m""")

    # Column used for predictions
    target_column = "Study_Hours_per_Day"

    # Read in CSV data
    df = read_csv(input_dir)
    print(f"\n\033[91mPrinting the first 10 rows of selected CSV:\033[00m")
    print(df.head(10))

    # Scan CSV to create dictionary of meta-data
    df_metadata = scan_df(df, target_column)
    print(f"\n\033[93mPrinting Meta-data that will be sent to LLM:\033[00m")
    print(json.dumps(df_metadata, indent= 4))

    # Initilise the results validator
    model_start = time.time()
    print(f"\n\033[94mStarting recommendation generation\033[00m")
    validated_json = result_validator(df, {}, target_column, df_metadata)
    final_plan, attempts = validated_json.run_validator(model_function, max_ai_calls)
    model_end = time.time() - model_start
    print(f"\n** Model took {model_end:.2f} seconds to generate results **")
    print(f"\n\033[94mPrinting final model recomendations:\033[00m")
    print(json.dumps(final_plan, indent= 4))
    print(f"\n Model validated dataset with {attempts} attempts")

    return None

if __name__ == "__main__":
    main()