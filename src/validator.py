import json
import pandas as pd
from typing import Dict, Any
from ai_call import call_lm_statistical
from sklearn.ensemble import IsolationForest
from sklearn.utils import all_estimators
from sklearn.utils.multiclass import type_of_target
from scipy.stats import skew

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

    def run_validator(self, max_attempts: int = None):
        '''Uses an iteritive approach to run through all validation checks up to maximum attempts and return validated report'''
        # Assing conditional variables
        attempts = 0
        feedback = None

        # Call LLM calls to check data iteration
        while attempts < max_attempts:
            output_json = call_lm_statistical(self.metadata, feedback)
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

        # Validate model is correct type
        task_validation = self.model_validator()
        if task_validation['status'] == 'invalid':
            all_errors.append(task_validation['errors'])

        # Validate outliers removal has been assigned correctly
        outlier_validation = self.outlier_validator()
        if outlier_validation['status'] == 'invalid':
            all_errors.append(outlier_validation['errors'])

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

        # Find the outlier removal suggestion   from recomendations and validate
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
                                \t Statistical evidence is too weak to justify removal.'''
                            )

        # Collect errors and add to list
        if errors:
            return {'status' : 'invalid', 'errors' : ' | '.join(errors)}
        return {'status' : 'valid'}

