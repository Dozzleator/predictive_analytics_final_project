import json
import time
from data_scan import read_csv, scan_df
from ai_call import call_lm_statistical

def main() -> None:
    # import specifiedx data and put into csv file
    df = read_csv("student_dropout_dataset.csv")
    print()
    print(df.head(10))

    df_metadata = scan_df(df, "Study_Hours_per_Day")
    print()
    print(json.dumps(df_metadata, indent= 4))

    model_start = time.time()
    json_ouput = call_lm_statistical(df_metadata)
    print(json_ouput)
    model_end = time.time() - model_start
    print(f"\nmodel took {model_end:.2f} seconds to generate results")

    return None

if __name__ == "__main__":
    main()