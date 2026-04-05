import optuna
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

def safe_expm1(x):
    """Safely exponentiates arrays, capping values to prevent float64 overflow."""
    x_float = np.asarray(x, dtype=np.float64)
    return np.expm1(np.clip(x_float, -700, 700))

def safe_sinh(x):
    """Safely applies inverse hyperbolic sine, capping values to prevent overflow."""
    x_float = np.asarray(x, dtype=np.float64)
    return np.sinh(np.clip(x_float, -700, 700))

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
            base_model = RandomForestRegressor(
                max_depth=trial.suggest_int('max_depth', 3, 10),
                n_jobs=-1
            )
        elif model_type == 'linear_regression':
            base_model = LinearRegression()
        else:
            base_model = LinearRegression()

        # Let optuna dynamically test if Log transformation is required for non-linear distributions only if values are positive
        use_target_tranform = trial.suggest_categorical('use_target_tranform', [True, False])

        if use_target_tranform:
            if y.min() >= 0:
                transform_func = np.log1p
                inverse_func = safe_expm1
                transform_name = 'log1p'
            else:
                transform_func = np.arcsinh
                inverse_func = safe_sinh
                transform_name = 'arcsinh'

            model = TransformedTargetRegressor(
                regressor=base_model,
                func=transform_func,
                inverse_func=inverse_func,
                check_inverse=False
            )
        else:
            model = base_model
            transform_name = 'none'

    # Eveluate pipeline results
    pipeline = Pipeline([('prep', preproccesor), ('model', model)])
    scoring = 'accuracy' if is_classification else 'r2'
    score = cross_val_score(pipeline, x, y, cv=KFold(n_splits=5), scoring=scoring).mean()

    # set transformation name in Optuna
    trial.set_user_attr('transform_name', transform_name)

    return score

def optimal_pipeline(df: pd.DataFrame, target_col: str, meta_data: dict, config: dict, target_accuracy: int, max_trials: int) -> tuple[dict, pd.DataFrame]:
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

    # Pull variables for search (set by user)
    target_dec = float(target_accuracy / 100)
    trial_runs = 0

    # Optimise within the trial limit
    while trial_runs < max_trials:
        # Create study to find optimal pipeline
        trial = study.ask()

        # Build study for each variation per trial
        score = build_optuna_objective(trial, x, y, is_classification, config)

        # Get result and feed back to optimiser for subsequent runs
        study.tell(trial, score)

        # Stop program if runs exced maximum trial runs (stops infinant looping)
        trial_runs += 1
        if len(study.trials) > 0 and study.best_value >= target_dec: 
            break

    # find and sort the completed trials
    trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    trials.sort(key=lambda t: t.value, reverse=True)

    # Build top three recommendations for global selections
    full_recommendation = {
        'target_column' : target_col,
        'task_type' : 'classification' if is_classification else 'regression',
        'pipeline_options' : []
    }

    # tracking_data list to build the metadata_df
    tracking_list = []

    # Seperate column by type
    num_cols = [col for col in x.columns if np.issubdtype(x[col].dtype, np.number)]
    cat_cols = [col for col in x.columns if not np.issubdtype(x[col].dtype, np.number)]

    # convert list into dict lookup
    col_lookup = {}
    for col_info in meta_data.get('columns', []):
        col_name = col_info.get('name')
        if col_name:
            col_lookup[col_name] = col_info

    # Create results and append to dict building the pipelines by top 3 ranking
    for rank, t in enumerate(trials[:3], start=1):
        params = t.params
        score = t.value

        # Find if column is actually missing data 
        num_cols_missing = [c for c in num_cols if col_lookup.get(c, {}).get('missing_values', 0) > 0]
        num_cols_clean = [c for c in num_cols if c not in num_cols_missing]
        
        cat_cols_missing = [c for c in cat_cols if col_lookup.get(c, {}).get('missing_values', 0) > 0]
        cat_cols_clean = [c for c in cat_cols if c not in cat_cols_missing]

        # Buid dynamic strategy list for numerical columns
        num_impute_strat = []
        if num_cols_missing: 
            num_impute_strat.append({
                'strategy' : params['num_impute'],
                'columns' : num_cols_missing
            })
        if num_cols_clean:
            num_impute_strat.append({
                'strategy' : 'none',
                'columns' : num_cols_clean
            })

        # Buid dynamic strategy list for categorical columns
        cat_impute_strat = []
        if cat_cols_missing: 
            cat_impute_strat.append({
                'strategy' : 'most_frequent',
                'columns' : cat_cols_missing
            })
        if cat_cols_clean:
            cat_impute_strat.append({
                'strategy' : 'none',
                'columns' : cat_cols_clean
            })

        # Check if log_trasform happen
        target_transformed = params.get('use_log_target', False)

        # pull transformation name saved to memory in Optuna at end of build func
        specific_transform = t.user_attrs.get('transform_name', 'none')

        # build the output dictionary
        pipeline_config = {
            'rank' : rank,
            'score': f'{score:.4f}',
            'model' : params['model_type'].replace('_', ' ').title(),
            'model_selection_justification' : '',
            'distribution_transformed' : target_transformed,
            'transformation_used': specific_transform,
            'distrabution_transformation_justification' : '',
            'transformations' : [
                {
                    'feature_type' : 'numeric',
                    'imputation' : num_impute_strat,
                    'scaling' : [{
                        'strategy' : params['num_scale'],
                        'columns' : num_cols,
                        'justification' : ''
                    }] if num_cols else [],
                    'encoding' : [{
                        'strategy' : 'none',
                        'columns' : num_cols,
                        'justification' : ''
                    }] if num_cols else [],
                },
                {
                    'feature_type' : 'categorical',
                    'imputation' : cat_impute_strat,
                    'scaling' : [{
                        'strategy' : 'none',
                        'columns' : cat_cols
                    }] if cat_cols else [],
                    'encoding' : [{
                        'strategy' : params['cat_encode'],
                        'columns' : cat_cols,
                        'justification' : ''
                    }] if cat_cols else [],
                },
            ],
        }

        # Append recommendations back to dict
        full_recommendation['pipeline_options'].append(pipeline_config)

        # Build into flat table for comprehenssion
        for col in x.columns:
            is_num = np.issubdtype(x[col].dtype, np.number)
            has_missing = col_lookup.get(col, {}).get('missing_values', 0) > 0

            # Look for actual applied transformations 
            if is_num:
                actual_impute = params['num_impute'] if has_missing else 'none'
                col_scale = params['num_scale']
                col_encode = 'none'
            else: 
                actual_impute = 'most_frequent' if has_missing else 'none'
                col_scale = 'none'
                col_encode = params['cat_encode']

            tracking_list.append({
                'Rank' : rank,
                'Column' : col,
                'Score' : f'{score:.4f}',
                'Imputation' : actual_impute,
                'Scaling' : col_scale,
                'Encoding' : col_encode,
                'Model' : params['model_type'].replace('_', ' ').title()
            })

    metadata_df = pd.DataFrame(tracking_list)
    return full_recommendation, metadata_df
