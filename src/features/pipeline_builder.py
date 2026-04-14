import optuna
import numpy as np
import pandas as pd
import warnings
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.exceptions import ConvergenceWarning

# Block these warnings from streamlit
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', module='sklearn')
np.seterr(all='ignore')

def safe_expm1(x):
    """Safely exponentiates arrays, capping values to prevent float64 overflow."""
    x_float = np.asarray(x, dtype=np.float64)
    return np.expm1(np.clip(x_float, -150, 150))

def safe_sinh(x):
    """Safely applies inverse hyperbolic sine, capping values to prevent overflow."""
    x_float = np.asarray(x, dtype=np.float64)
    return np.sinh(np.clip(x_float, -150, 150))

def build_optuna_objective(trial: optuna.Trial, x: pd.DataFrame, y: pd.DataFrame, is_classification: bool, config: dict) -> float:
    '''Build and test to find optimal pipeline (Baysian search)'''
    impute_strat = config.get('imputation', [])
    scaling_strat = config.get('scaling',[])
    encode_strat = config.get('encoding',[])

    models_dict = config.get('models', {})
    class_models = models_dict.get('classification', [])
    reg_models = models_dict.get('regression', [])

    # 1. Ask Optuna to select the algorithm FIRST
    if is_classification:
        model_type = trial.suggest_categorical('model_type', class_models)
    else:
        model_type = trial.suggest_categorical('model_type', reg_models)

    # 2. Ask Optuna to select the preprocessing steps
    num_impute = trial.suggest_categorical('num_impute', impute_strat)
    num_scale = trial.suggest_categorical('num_scale', scaling_strat)
    cat_encode = trial.suggest_categorical('cat_encode', encode_strat)

    # 3. The Neural Network Survival Override
    if model_type == 'neural_network' and num_scale == 'none':
        num_scale = 'standard'

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

    use_early_stopping = len(x) >= 50
    model_max_iter = 800

    if is_classification:
        # Random Forrest Classifier
        if model_type == 'random_forest':
            model = RandomForestClassifier(max_depth=trial.suggest_int('max_depth', 3, 10), n_jobs=-1)

        # MLP Neural Network Classifier
        elif model_type == 'neural_network':
            model = MLPClassifier(
                solver='lbfgs',
                hidden_layer_sizes=(trial.suggest_int('mlp_c_layer_1', 10, 50), trial.suggest_int('mlp_c_layer_2', 5, 20)),
                activation=trial.suggest_categorical('mlp_c_activation', ['relu', 'tanh']),
                alpha=trial.suggest_float('mlp_c_alpha', 1e-4, 1e-1, log=True),
                max_iter=model_max_iter,
                early_stopping=use_early_stopping
            )
        else:
            model = LogisticRegression(max_iter=model_max_iter)

        # Set this as classification problems should not have log transformations
        transform_name = 'none'

    else:
        # Random Forrest Regression
        if model_type == 'random_forest':
            base_model = RandomForestRegressor(max_depth=trial.suggest_int('max_depth', 3, 10), n_jobs=-1)

        # Linear Regression
        elif model_type == 'linear_regression':
            base_model = LinearRegression()

        # Polynomial Regression
        elif model_type == 'polynomial_regression':
            poly_degree = trial.suggest_int('poly_degree', 2, 3)
            base_model = make_pipeline(PolynomialFeatures(degree=poly_degree, include_bias=False), LinearRegression())

        # MLP Neural Network Regression
        elif model_type == 'neural_network':
            base_model = MLPRegressor(
                solver='lbfgs',
                hidden_layer_sizes=(trial.suggest_int('mlp_r_layer_1', 10, 50), trial.suggest_int('mlp_r_layer_2', 5, 20)),
                activation=trial.suggest_categorical('mlp_r_activation', ['relu', 'tanh']),
                alpha=trial.suggest_float('mlp_r_alpha', 1e-4, 1e-1, log=True),
                max_iter=model_max_iter,
                early_stopping=use_early_stopping
            )
        else:
            base_model = LinearRegression()

        use_target_transform = trial.suggest_categorical('use_target_transform', [True, False])

        if use_target_transform:
            if y.min() >= 0:
                transform_func = np.log1p
                inverse_func = safe_expm1
                transform_name = 'log1p'
            else:
                transform_func = np.arcsinh
                inverse_func = safe_sinh
                transform_name = 'arcsinh'

            model = TransformedTargetRegressor(regressor=base_model, func=transform_func, inverse_func=inverse_func, check_inverse=False)
        else:
            model = base_model
            transform_name = 'none'

    # Evaluate pipeline results
    pipeline = Pipeline([('prep', preproccesor), ('model', model)])
    scoring = 'accuracy' if is_classification else 'r2'

    # Define the validation strategy
    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

    score = cross_val_score(pipeline, x, y, cv=cv_strategy, scoring=scoring).mean()
    trial.set_user_attr('transform_name', transform_name)

    return score

def optimal_pipeline(df: pd.DataFrame, target_col: str, meta_data: dict, config: dict, max_trials: int) -> tuple[dict, pd.DataFrame]:
    '''Automatically builds pipelines based on first prompted suggestions'''
    
    # Drop rows that have null values on the predictor column
    df = df.dropna(subset=[target_col])

    # Automatic collinearity transformation
    dropped_cols = []
    num_df = df.select_dtypes(include=[np.number])

    if not num_df.empty:
        # Calculate absolute correlation of features
        corr_matrix = num_df.corr().abs()

        # Get upper triangle of data so same value is not selected twice
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find columns with more then 90% correlation with other columns
        dropped_cols = [col for col in upper.columns if any(upper[col] > 0.90) and col != target_col]

        if dropped_cols:
            df = df.drop(columns=dropped_cols)

    # Guarantee chronological order so TimeSeriesSplit doesn't read backwards
    time_cols = [col for col in df.columns if 'year' in col.lower() or 'date' in col.lower()]
    if time_cols:
        df = df.sort_values(time_cols[0], ascending=True).reset_index(drop=True)

    # Reduce dataset if it is large to reduce operational effeciency
    if len(df) > 1000:
        df = df.sample(n=1000, random_state=42)

    # find x and y variables
    x = df.drop(columns=[target_col])
    y = df[target_col]

    # find if model is classification type from metadata
    is_classification = 'classification' in meta_data.get('model_task_type', '').lower()

    # remove warning and set goal
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')

    # get model and find solver type
    model_dict = config.get('models', {})
    available_models = model_dict.get('classification', []) if is_classification else model_dict.get('regression', [])

    # run through all models iterativly
    for model_name in available_models:
        study.enqueue_trial({'model_type': model_name})

    # end optimisation when max trials reached
    trial_runs = 0

    while trial_runs < max_trials:
        trial = study.ask()
        score = build_optuna_objective(trial, x, y, is_classification, config)
        study.tell(trial, score)

        trial_runs += 1

    # get results from optuna optimisation
    trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    trials.sort(key=lambda t: t.value, reverse=True)

    # collect data without storing duplicates
    unique_trials = []
    seen_scores = set()
    
    for t in trials:
        rounded_score = round(t.value, 4)
        if rounded_score not in seen_scores:
            unique_trials.append(t)
            seen_scores.add(rounded_score)
        if len(unique_trials) == 3:
            break

    # build pipeline recommendation meta_data
    full_recommendation = {
        'target_column' : target_col,
        'task_type' : 'classification' if is_classification else 'regression',
        'pipeline_options' : []
    }

    tracking_list = []

    num_cols = [col for col in x.columns if np.issubdtype(x[col].dtype, np.number)]
    cat_cols = [col for col in x.columns if not np.issubdtype(x[col].dtype, np.number)]

    # create column dictionary to look through arr of results
    col_lookup = {}
    for col_info in meta_data.get('columns', []):
        col_name = col_info.get('name')
        if col_name:
            col_lookup[col_name] = col_info

    # loop through top three rank of runs
    for rank, t in enumerate(unique_trials, start=1):
        params = t.params
        score = t.value

        num_cols_missing = [c for c in num_cols if col_lookup.get(c, {}).get('missing_values', 0) > 0]
        num_cols_clean = [c for c in num_cols if c not in num_cols_missing]
        
        cat_cols_missing = [c for c in cat_cols if col_lookup.get(c, {}).get('missing_values', 0) > 0]
        cat_cols_clean = [c for c in cat_cols if c not in cat_cols_missing]

        # find imput strategy
        num_impute_strat = []
        if num_cols_missing: 
            num_impute_strat.append({'strategy' : params['num_impute'], 'columns' : num_cols_missing})
        if num_cols_clean:
            num_impute_strat.append({'strategy' : 'none', 'columns' : num_cols_clean})

        cat_impute_strat = []
        if cat_cols_missing: 
            cat_impute_strat.append({'strategy' : 'most_frequent', 'columns' : cat_cols_missing})
        if cat_cols_clean:
            cat_impute_strat.append({'strategy' : 'none', 'columns' : cat_cols_clean})

        target_transformed = params.get('use_target_transform', False)
        specific_transform = t.user_attrs.get('transform_name', 'none')

        # build pipeline dictionary (needed to log transformations and model selection)
        pipeline_config = {
            'rank' : rank,
            'score': f'{score:.4f}',
            'model' : params['model_type'].replace('_', ' ').title(),
            'model_selection_justification' : '',
            'distribution_transformed' : target_transformed,
            'transformation_used': specific_transform,
            'distribution_transformation_justification' : '',
            'transformations' : []
        }

        # update if dataframe modefied with colinearity filter 
        if dropped_cols:
            pipeline_config['transformations'].append({
                'feature_type': 'numeric',
                'feature_selection': [{
                    'strategy': 'collinearity_filter',
                    'columns': dropped_cols,
                    'justification': ''  
                }]
            })

        # Append the other transformations to the pipeline (imputation, scaling, encoding)
        pipeline_config['transformations'].extend([
            {
                'feature_type': 'numeric',
                'imputation': num_impute_strat,
                'scaling': [{
                    'strategy': params['num_scale'], 
                    'columns': num_cols, 
                    'justification': ''
                }] if num_cols else [],
                'encoding': [{
                    'strategy': 'none', 
                    'columns': num_cols, 
                    'justification': ''
                }] if num_cols else [],
            },
            {
                'feature_type': 'categorical',
                'imputation': cat_impute_strat,
                'scaling': [{
                    'strategy': 'none', 
                    'columns': cat_cols,
                    'justification': ''
                }] if cat_cols else [],
                'encoding': [{
                    'strategy': params['cat_encode'], 
                    'columns': cat_cols, 
                    'justification': ''
                }] if cat_cols else [],
            }
        ])

        full_recommendation['pipeline_options'].append(pipeline_config)

        for col in x.columns:
            is_num = np.issubdtype(x[col].dtype, np.number)
            has_missing = col_lookup.get(col, {}).get('missing_values', 0) > 0

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