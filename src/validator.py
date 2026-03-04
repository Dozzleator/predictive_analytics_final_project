import json
import pandas as pd
from typing import Dict, Any
from sklearn.ensemble import IsolationForest
from sklearn.utils import all_estimators
from sklearn.utils.multiclass import type_of_target

class result_validator:
    def __init__(self, df: pd.DataFrame, ai_result: Dict[str, Any], target_col: Any, meta_data: str):
        self.df = df
        self.metadata = meta_data
        self.ai_result = ai_result
        self.target_col = target_col
        self.validation_report = []

        # Load in all sk-learn regression and classification models for validation
        self.regression_models = [name for name, _ in all_estimators(type_filter='regressor')]
        self.classification_models = [name for name, _ in all_estimators(type_filter='classifier')]

    def run_validator(self, model_function, max_attempts: int = None):
        '''Uses an iteritive approach to run through all validation checks up to maximum attempts and return validated report'''
        # Assing conditional variables
        attempts = 0
        feedback = None

        # Call LLM calls to check data iteration
        while attempts < max_attempts:
            output_json = model_function(self.metadata, feedback)
            self.ai_result = json.loads(output_json)

            # go through all checks to generate a final report
            final_validated_report = self.run_all_checks()

            # If there are no invalid the report is food (break loop)
            if final_validated_report['status'] == 'valid':
                attempts += 1
                return self.ai_result, attempts

            # If invalid prepare feedback to send to model for next round
            feedback = final_validated_report['errors']
            attempts += 1

        return self.ai_result, attempts

    def run_all_checks(self) -> Dict[str, Any]:
        '''function to run through all the validation checks and put results into single report'''
        # List to hold all error information
        all_errors = []

        # Add validation check functions to list
        check_methods = [
                    self.model_validator,
                    self.outlier_validator,
                    self.cardinality_validator,
                    self.imputation_validator,
                    self.scaling_validator,
                    self.multicollinearity_validator
                ]

        # Run checks and if error raised add to list
        for method in check_methods:
            result = method()
            if result['status'] == 'invalid':
                all_errors.append(result['errors'])

        if all_errors:
            return {'status' : 'invalid', 'errors': ' | '.join(all_errors)}
        return {'status' : 'valid'}

    def model_validator(self) -> Dict[str, Any]:
        """Using SKLearn verify that the correct model type has been selected (Classification vs Regression)"""
        # Identify the target classification type using sklearn
        model_type = type_of_target(self.df[self.target_col].dropna())

        # collect errors to return to LLM in batch
        errors = []

        # Find the model selection from recomendations and validate
        for strategy in self.ai_result.get('global_strategies', []):
            if strategy.get('category', '').lower() in ['modeling', 'model selection']:
                for opt in strategy.get('top_three_options', []):
                        model_name_full = opt['name'].lower()
                        model_name = model_name_full.replace('regressor', '').replace('classifier', '').strip()

                        # model lookup to find classification & regression models in sklearn
                        is_classification = any(model_name in c.lower() for c in self.classification_models)
                        is_regression = any(model_name in r.lower() for r in self.regression_models)

                        # Check model is valid
                        if model_type == 'continuous' and 'classifier' in model_name_full:
                            errors.append(f"Named a classification model ({model_name_full}) for a regression task")
                        elif model_type != 'continuous' and 'regressor' in model_name_full:
                            errors.append(f"Named a regression model ({model_name_full}) for a classification task")

                        # Check suggested model is valid against SK-Learn models and is in correct category
                        elif model_type == 'continuous' and (is_classification and not is_regression): 
                            errors.append(f'The {model_name_full} model is not typically associated with regression problems')
                        elif model_type != 'continuous' and (is_regression and not is_classification): 
                            errors.append(f'The {model_name_full} model is not typically associated with classification problems')
                        elif not is_regression  and not is_classification:
                            errors.append(f'The {model_name_full} model is not part of the sk-learn model library')

        # Collect errors and add to list
        if errors:
            return {'status' : 'invalid', 'errors' : ' | '.join(errors)}
        return {'status' : 'valid'}

    def outlier_validator(self) -> Dict[str, Any]:
        '''Using Isolation Forrest from scikit-learn validate LLM outlier removal suggestion'''
        # collect errors to return to LLM in batch
        errors = []

        # Find the outlier removal suggestion from recomendations and validate
        for strategy in self.ai_result.get('column_strategies', []):
            column_name = strategy['column_name']
            if column_name in self.df.columns and pd.api.types.is_numeric_dtype(self.df[column_name]):
                options = strategy.get('top_three_options', [])

                # Check that outlier removal was suggested before validating
                if any('outlier' in opt.get('name', '').lower() for opt in options):
                    # statistical check if valid
                    iso = IsolationForest(contamination= 0.05, random_state= 42)
                    data = self.df[[column_name]].dropna()

                    # predict outliers if data exists 
                    if not data.empty:
                        prediction = iso.fit_predict(data)
                        outlier_count = (prediction == -1).sum()
                        threshold = len(data) * 0.05

                        # 3. Throw outlier error if there is less then 5% outliers and outlier removal was suggested
                        if outlier_count < threshold:
                            errors.append(
                                f'''Outlier removal suggested for {column_name}, but only {outlier_count} 
                                \t outliers ({outlier_count/len(data):.1%}) were detected. 
                                \t Statistical evidence is too weak to justify removal.'''.strip()
                            )

        # Collect errors and add to list
        if errors:
            return {'status' : 'invalid', 'errors' : ' | '.join(errors)}
        return {'status' : 'valid'}

    def cardinality_validator(self) -> Dict[str, Any]:
        '''Validate that encoding recommendations'''
        # collect errors to return to LLM in batch
        errors = []

        # Map the column names to associated meta_data (generated in data_scan.py)
        meta_lookup = {col['name']: col for col in self.metadata.get('columns', [])}

        # Find column information for validation check
        for strategy in self.ai_result.get('column_strategies', []):
            column_name = strategy['column_name']
            column_data = meta_lookup.get(column_name)

            # continue search if no meta_data found in column
            if not column_data:
                continue

            # Find top three results suggested by AI
            options = strategy.get('top_three_options', []) 

            # High cardinality validation (don't use One-Hot encoding)
            if column_data.get('high_cardinality') is True:
                if any('one' in opt.get('name', '').lower() for opt in options):
                    errors.append(
                        f'''Column {column_name} has high cardinality ({column_data['unique_values']} values)
                        \t One-hot Encoding may lead to overfiting suggest the use of target encoding or feature hashing instead'''.strip()
                    )

            # Check datatype to make sure encoding is plausable
            if column_data.get('is_categorical') is False and column_data.get('type') != 'str':
                if any('encod' in opt.get('name', '').lower() for opt in options):
                    errors.append(
                        f'''Encoding suggested for {column_name} column
                        \thowever column metadata shows dtype is already numeric {column_data['type']}, suggest scalling data instead'''.strip()
                    )

            # Check if binary (Encoding not required)
            if column_data.get('is_binary') is True:
                if any(term in opt.get('name', '').lower() for opt in options for term in ['target', 'one']):
                    errors.append(
                        f'''Column {column_name} is binary data. Encoding is not required
                        \tSuggest binary mapping'''
                    )

            # Check for ordinal encoding hallucinations
            if any('ordinal' in opt.get('name', '').lower() for opt in options):
                if column_data.get('is_categorical') is False:
                    errors.append(f'Ordinal encoding was suggested for {column_name}, but is is a continuous numeric column'.strip())
                if column_data.get('unique_values', 0) > 10:
                    errors.append(
                        f'''Ordinal encoding suggested for {column_name} but has {column_data['unique_values']} unique values
                        \tOrdinal encoding is generally used for small ranked sets (eg. low, medium, high)'''.strip()
                    )

        # Collect errors and add to list    
        if errors:
            return {'status' : 'invalid', 'errors' : ' | '.join(errors)}
        return {'status' : 'valid'}

    def imputation_validator(self) -> Dict[str, Any]:
        '''Ensure imputation is only recommended on columns with missing values'''
        # collect errors to return to LLM in batch
        errors = []

        # Map the column names to associated meta_data (generated in data_scan.py)
        meta_lookup = {col['name']: col for col in self.metadata.get('columns', [])}

        # Find column information for validation check
        for strategy in self.ai_result.get('column_strategies', []):
            column_name = strategy['column_name']
            column_data = meta_lookup.get(column_name)

            # continue search if no meta_data found in column
            if not column_data:
                continue

            # Find top three results suggested by AI
            options = strategy.get('top_three_options', []) 

            # Check for missing values 
            has_missing = column_data.get('missing_values', 0) > 0

            # Validate reccomendation for imputation or filling is not on empty columns
            if any(term in opt.get('name', '').lower() for opt in options for term in ['imput', 'fill']):
                if not has_missing:
                    errors.append(
                        f'''Imputation suggested for {column_name} but does not have missing values
                        \tsuggest scalling or transformation'''.strip()
                    )

        # Collect errors and add to list    
        if errors:
            return {'status' : 'invalid', 'errors' : ' | '.join(errors)}
        return {'status' : 'valid'}

    def scaling_validator(self) -> Dict[str, Any]:
        '''Check standardisation recommendations made by LLM and validate legitimacy'''
        # collect errors to return to LLM in batch
        errors = []

        # Map the column names to associated meta_data (generated in data_scan.py)
        meta_lookup = {col['name']: col for col in self.metadata.get('columns', [])}

        # Find column information for validation check
        for strategy in self.ai_result.get('column_strategies', []):
            column_name = strategy['column_name']
            column_data = meta_lookup.get(column_name)

            # continue search if no meta_data or numeric data found in column
            if not column_data or not pd.api.types.is_numeric_dtype(self.df):
                continue

            # Find top three results suggested by AI
            options = strategy.get('top_three_options', []) 

            # Find data skew for each column
            skewness = column_data.get('skewness', 0)

            # Validate logrithmic transformation suggestion (make sure data has skew)
            if any('log' in opt.get('name', '').lower() for opt in options):
                if abs(skewness) < 0.5:
                    errors.append(
                        f'''Log transformation suggested for {column_name} but skewness for the column is only {skewness:.2f}
                        \tData is nearly normally suggest standard scalling instead'''.strip()
                    )

            # Validate standard scaling suggestion (make sure not highly skewed)
            if any('standard' in opt.get('name', '').lower() for opt in options):
                if abs(skewness) > 1.5:
                    errors.append(
                        f'''Standardisation suggested for {column_name} but column is highly skewed {skewness:.2f}
                        \tSuggest another standardisation technique to handle highly skewed data instead'''.strip()
                    )

        # Collect errors and add to list    
        if errors:
            return {'status' : 'invalid', 'errors' : ' | '.join(errors)}
        return {'status' : 'valid'}

    def multicollinearity_validator(self) -> Dict[str, Any]:
        '''Check to see if AI is accuratly reducing or keeping features'''
        # collect errors to return to LLM in batch
        errors = []

        # get meta_data of correlation related to columns
        correlated_columns = self.metadata.get('high_corilation_features', [])

        # end search if there are no highly correlated columns
        if not correlated_columns:
            return {'status': 'valid'}

        # Find column names for columns that have had AI suggestions
        suggested_names = [strategy.get('column_name') for strategy in self.ai_result.get('column_strategies', [])]

        # find only the columns where correlation has actually been recommended
        recommendation_columns = [col for col in correlated_columns if col in suggested_names]

        # If there are highly correlated columns then rais error
        if len(recommendation_columns) > 1:
            errors.append(
                f'''Features in {recommendation_columns} are highly correlated, Using all will introduce multicollinearity
                \tSuggest dropping redunted features'''.strip()
            )

        # Collect errors and add to list    
        if errors:
            return {'status' : 'invalid', 'errors' : ' | '.join(errors)}
        return {'status' : 'valid'}