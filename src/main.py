import pandas as pd
import numpy as np
import json
from pathlib import Path

def read_csv(data_name: str) -> pd.DataFrame:
    '''Import CSV data from the input data file'''

    #Get location of current script
    current_dir = Path(__file__).parent

    # Find path to wanted csv file
    csv_data = current_dir.parent / f"input_data/{data_name}"

    # load CSV data in pandas dataframe
    df = pd.read_csv(csv_data)

    return df

def identify_task_type(df: pd.DataFrame, predict_col: str = None) -> str:
    '''Find what model type should be considered for dataset'''
    # set to unkown if cannot find prediction column
    if not predict_col or predict_col not in df.columns:
        return "unkown"

    # If datatype mostly text return classification
    if not pd.api.types.is_numeric_dtype(df[predict_col]):
        return "classification"

    # If numeric with few unique values return classification
    if df[predict_col].nunique() <= 10:
        return "classification"

    else:
        return "regression"

def scan_df(df: pd.DataFrame, predict_col: str = None) -> dict:
    '''Scan dataframe and extract statistical metadata'''

    # run function to find model type
    task_type = identify_task_type(df, predict_col)

    # store general information related to the dataframe
    metadata = {
        "total_rows": len(df),
        "prediction_variable": predict_col,
        "model_task_type": task_type,
        "columns": [],
        "high_corilation_features": [],
    }

    # store datascience information related to each column
    for col in df.columns:
        # Find cardinality ratio to identify if column is usefull
        cardinality_ratio = round(df[col].nunique() / len(df), 4)

        # Find unique values in each column helps to identify regression type
        unique_values = int(df[col].nunique())

        col_info = {
            "name": col,
            "type": str(df[col].dtype),
            "is_target": col == predict_col,
            "missing_percentage" : round((int(df[col].isnull().sum()) / len(df)) * 100, 2),
            "missing_values": int(df[col].isnull().sum()),
            "unique_values": unique_values,
            "cardinality_ratio": cardinality_ratio,
            "high_cardinality": cardinality_ratio > 0.5 and unique_values > 20 and col != predict_col,
            "is_categorical": unique_values <= 10 or not pd.api.types.is_numeric_dtype(df[col]),
            "is_binary": unique_values == 2
        }

        # identify if column has unique values (is | not constant)
        col_info["is_constant"] = col_info["unique_values"] == 1

        # identify and store datascience metrics
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["skewness"] = round(df[col].skew(), 2)
            col_info["mean"] = round(df[col].mean(), 2)

            # identify outliers for noise reduction
            q1 = df[col].quantile(0.25) 
            q3 = df[col].quantile(0.75) 
            iqr = q3 - q1
            outliers = df[(df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))]
            col_info["outlier_count"] = len(outliers)
            col_info["has_outliers"] = len(outliers) > 0

        # append column information to the empty list in "metadata"
        metadata["columns"].append(col_info)

    # correlation (Multicollinearity)
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df) > 2: 
        correlation_matrix = df.select_dtypes(include=[np.number]).corr().abs()
        upper_bound = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        high_correlation = [column for column in upper_bound.columns if any(upper_bound[column] > 0.90)]
        metadata["high_corilation_features"] = high_correlation

    return metadata

def main() -> None:
    # import specifiedx data and put into csv file
    df = read_csv("student_dropout_dataset.csv")
    print()
    print(df.head(10))

    df_metadata = scan_df(df, "Study_Hours_per_Day")
    print()
    print(json.dumps(df_metadata, indent= 4))

    return None


if __name__ == "__main__":
    main()