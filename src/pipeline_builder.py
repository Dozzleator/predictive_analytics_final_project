import optuna
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

def build_optuna_objective(trial: optuna.Trial, x: pd.DataFrame, y: pd.DataFrame, is_classification: bool, config: dict) -> float:
    '''Build and test to find optimal pipeline (Baysian search)'''
    # Get list data from config file
    impute_strat = config.get('imputation', [])
    scaling_strat = config.get('scaling',[])
    encode_strat = config.get('encoding',[])

    # Define search space (Preprocessing requirements)
    num_impute = trial.suggest_categorical('num_impute', impute_strat)
    num_scale = trial.suggest_categorical('num_scale', scaling_strat)
    cat_encode = trial.suggest_categorical('cat_encode', encode_strat)

    # Map strings to functional strategies
    imputer_map = {imp: SimpleImputer(strategy=imp) for imp in impute_strat}
    scaler_map = {'standard': StandardScaler(), 'robust': RobustScaler(), 'none': 'passthrough'}
    encode_map = {'one_hot': OneHotEncoder(handle_unknown='ignore', sparse_output=False), 'ordinal': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)}

    # Build Transformer for numerical columns
    num_steps = [('impute', imputer_map[num_impute])]
    if num_scale != 'none':
        num_steps.append(('scale', scaler_map[num_scale]))

    # Build Transformer for categorical columns
    cat_steps = [
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', encode_map[cat_encode])
    ]

    # Feed processor columns of correct dtypes
    preproccesor = ColumnTransformer([
        ('num', Pipeline(num_steps), make_column_selector(dtype_include=np.number)),
        ('cat', Pipeline(cat_steps), make_column_selector(dtype_exclude=np.number))
    ])

    # get model names from the config
    models_dict = config.get('models', {})
    class_models = models_dict.get('classification', [])
    reg_models = models_dict.get('regression', [])

    # Sparse search setup / model selection for classification models
    if is_classification:
        model_type = trial.suggest_categorical('model_type', class_models)
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                max_depth=trial.suggest_int('max_depth', 3, 10),
                n_jobs=-1
            )
        else:
            model = LogisticRegression(max_iter=1000)
    # For regression models
    else:
        model_type = trial.suggest_categorical('model_type', reg_models)
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                max_depth=trial.suggest_int('max_depth', 3, 10),
                n_jobs=-1
            )
        else:
            model = LinearRegression()

    # Eveluate pipeline results
    pipeline = Pipeline([('prep', preproccesor), ('model', model)])
    scoring = 'accuracy' if is_classification else 'r2'
    score = cross_val_score(pipeline, x, y, cv=KFold(n_splits=5), scoring=scoring).mean()

    return score

def optimal_pipeline(df: pd.DataFrame, target_col: str, meta_data: dict, config: dict, n_trials: int) -> tuple[float, dict, pd.DataFrame]:
    '''Automatically builds pipelines based on first prompted suggrstions'''

    # Remove rows where target col is N/A
    df = df.dropna(subset=[target_col])

    # Reduce size of dataset for improved performance
    if len(df) > 1000:
        df = df.sample(n=1000, random_state= 42)

    # Seperates Features and Target
    x = df.drop(columns=[target_col])
    y = df[target_col]

    # Identify and assign task types
    is_classification = 'classification' in meta_data.get('model_task_type', '').lower()

    # Run optuna optimiser
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')

    # pass vriavles into function with lambda
    study.optimize(lambda trial: build_optuna_objective(trial, x, y, is_classification, config), n_trials= n_trials)

    # find and sort the completed trials
    trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    trials.sort(key=lambda t: t.value, reverse=True)

    # Initialise the structured results 
    full_recommendation = {
        'global_strategies': [],
        'column_strategies': []
    }

# Extract unique models from top trials, limited to top 3
    unique_models = list(dict.fromkeys([t.params['model_type'] for t in trials]))[:3]

    # Build top three recommendations for global selections
    full_recommendation['global_strategies'].append({
        'category': 'Model Selection',
        'model_options': [m.replace('_', ' ').title() for m in unique_models],
        'cv_recommendations': 'kfold',
        'cv_folds': 5,
        'justification': ''
    })

    # tracking_data list to build the metadata_df
    tracking_list = []

    for col in x.columns:
        is_num = np.issubdtype(x[col].dtype, np.number)

        # build columnwise dictionary for top three recomendations
        col_rec = {
            'column_name': col,
            'imputation_options': [],
            'scaling_options': [],
            'encoding_options': [],
            'reasoning': ''
        }

        # Build the structured dict for the LLM and the tracking list for the UI table
        for t in trials[:3]:
            params = t.params
            score = t.value
            
            # Map parameters based on type
            col_impute = params['num_impute'] if is_num else 'most_frequent'
            col_scale = params['num_scale'] if is_num else 'none'
            col_encode = 'none' if is_num else params['cat_encode']

            col_rec['imputation_options'].append({'technique': col_impute, 'justification': ''})
            col_rec['scaling_options'].append({'technique': col_scale, 'justification': ''})
            col_rec['encoding_options'].append({'technique': col_encode, 'justification': ''})

            # Add to the flat tracking list for the metadata_df
            tracking_list.append({
                'Column': col,
                'Score': f'{score:.4f}',
                'Imputation': col_impute,
                'Scaling': col_scale,
                'Encoding': col_encode,
                'Model': params['model_type'].replace('_', ' ').title()
            })

        full_recommendation['column_strategies'].append(col_rec)

    metadata_df = pd.DataFrame(tracking_list)
    return full_recommendation, metadata_df
