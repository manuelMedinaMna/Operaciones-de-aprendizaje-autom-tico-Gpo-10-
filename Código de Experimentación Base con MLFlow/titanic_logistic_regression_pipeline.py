import argparse
import json
import os
import random
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault('MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR', 'false')

import matplotlib

matplotlib.use('Agg')  # backend sin ventana: el script solo guarda las figuras
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore', message='Hint: Inferred schema contains integer column')

TARGET = 'survived'

RAW_FEATURES = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'who', 'alone', 'deck']

NUM_FEATURES = ['pclass', 'age', 'sibsp', 'parch', 'fare_log']
CAT_FEATURES = ['sex', 'embarked', 'who']
BIN_FEATURES = ['alone', 'has_deck', 'age_missing']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Entrena una Regresión Logística sobre Titanic y registra la corrida en MLflow.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--C', type=float, default=1.0,
                        help='Inverso de la fuerza de regularización (menor = más regularización).')
    parser.add_argument('--l1-ratio', type=float, default=0.0,
                        help='Mezcla de la penalización: 0 = L2, 1 = L1, intermedio = Elastic-Net.')
    parser.add_argument('--class-weight', choices=['none', 'balanced'], default='none',
                        help='Ponderación de clases para compensar el desbalance.')
    parser.add_argument('--max-iter', type=int, default=1000,
                        help='Máximo de iteraciones del solver.')

    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla para la partición, la validación cruzada y el modelo.')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proporción del conjunto de prueba.')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Número de folds de la validación cruzada estratificada.')
    parser.add_argument('--eval-test', action='store_true',
                        help='Evalúa también en el conjunto de prueba (solo para el modelo final).')
    # MLflow
    parser.add_argument('--experiment-name', default='titanic-logreg-hiperparametros',
                        help='Experimento de MLflow que agrupa las corridas.')
    parser.add_argument('--tracking-uri', default='sqlite:///mlflow.db',
                        help='URI del backend de MLflow Tracking.')
    parser.add_argument('--run-name', default=None,
                        help='Nombre de la corrida (por defecto se arma con los hiperparámetros).')
    args = parser.parse_args(argv)
    if not 0.0 <= args.l1_ratio <= 1.0:
        parser.error('--l1-ratio debe estar entre 0 y 1')
    return args


def set_seed(seed):
    """Fija las semillas globales. Los componentes de scikit-learn reciben además `random_state` explícito."""
    random.seed(seed)
    np.random.seed(seed)


def load_data():
    """Carga Titanic desde seaborn y separa features crudas y target."""
    df = sns.load_dataset('titanic')
    X = df[RAW_FEATURES].copy()

    X['deck'] = X['deck'].astype(object)
    return X, df[TARGET]


def split_data(X, y, test_size, seed):
    """Partición estratificada antes de cualquier transformación, para evitar fuga de información."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def add_deterministic_features(X):
    """Transformaciones fila por fila, que no aprenden nada de los datos."""
    X = X.copy()
    X['has_deck'] = X['deck'].notna().astype(int)
    X['age_missing'] = X['age'].isna().astype(int)
    X['fare_log'] = np.log1p(X['fare'])
    X['alone'] = X['alone'].astype(int)
    return X.drop(columns=['deck', 'fare'])


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Imputa `column` con la mediana por grupo aprendida en `fit` (mediana global si el grupo no existe)."""

    def __init__(self, column='age', group_cols=('sex', 'pclass')):
        self.column = column
        self.group_cols = group_cols

    def fit(self, X, y=None):
        self.medians_ = X.groupby(list(self.group_cols), observed=True)[self.column].median()
        self.global_median_ = X[self.column].median()
        return self

    def transform(self, X):
        X = X.copy()
        idx = pd.MultiIndex.from_frame(X[list(self.group_cols)])
        por_grupo = pd.Series(self.medians_.reindex(idx).to_numpy(), index=X.index)
        X[self.column] = X[self.column].fillna(por_grupo).fillna(self.global_median_)
        return X


def resolve_solver(l1_ratio):
    return 'lbfgs' if l1_ratio == 0 else 'saga'


def build_pipeline(C, l1_ratio, class_weight, max_iter, seed):
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), NUM_FEATURES),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first')),
        ]), CAT_FEATURES),
        ('bin', 'passthrough', BIN_FEATURES),
    ])
    model = LogisticRegression(
        C=C,
        l1_ratio=l1_ratio,
        class_weight=None if class_weight == 'none' else class_weight,
        solver=resolve_solver(l1_ratio),
        max_iter=max_iter,
        random_state=seed,
    )
    return Pipeline([
        ('features', FunctionTransformer(add_deterministic_features)),
        ('age_imputer', GroupMedianImputer()),
        ('preprocessor', preprocessor),
        ('model', model),
    ])


def cross_validate_model(pipeline, X_train, y_train, cv):
    scoring = {
        'accuracy': 'accuracy',
        'f1': make_scorer(f1_score, zero_division=0),
        'pr_auc': 'average_precision',
    }
    scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)
    metrics = {}
    for name in scoring:
        metrics[f'{name}_train'] = float(scores[f'train_{name}'].mean())
        metrics[f'{name}_cv'] = float(scores[f'test_{name}'].mean())
        metrics[f'{name}_cv_std'] = float(scores[f'test_{name}'].std())
    return metrics


def evaluate_on_test(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        'accuracy_test': float(accuracy_score(y_test, y_pred)),
        'f1_test': float(f1_score(y_test, y_pred, zero_division=0)),
        'pr_auc_test': float(average_precision_score(y_test, y_proba)),
    }


def save_confusion_matrix(y_true, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=['No sobrevivió', 'Sobrevivió'], cmap='Blues', colorbar=False, ax=ax
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_coefficients(pipeline, path):
    coefs = pd.Series(
        pipeline.named_steps['model'].coef_[0],
        index=pipeline.named_steps['preprocessor'].get_feature_names_out(),
        name='coeficiente',
    ).sort_values()
    coefs.to_csv(path)


def default_run_name(args):
    return f'logreg_C{args.C:g}_l1{args.l1_ratio:g}_cw-{args.class_weight}_seed{args.seed}'


def run_experiment(args):
    set_seed(args.seed)

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, args.test_size, args.seed)

    pipeline = build_pipeline(args.C, args.l1_ratio, args.class_weight, args.max_iter, args.seed)
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    metrics = cross_validate_model(pipeline, X_train, y_train, cv)
    # Predicciones fuera de fold: matriz de confusión honesta sin tocar el conjunto de prueba
    y_oof = cross_val_predict(pipeline, X_train, y_train, cv=cv)

    pipeline.fit(X_train, y_train)
    if args.eval_test:
        metrics.update(evaluate_on_test(pipeline, X_test, y_test))

    params = {
        'C': args.C,
        'l1_ratio': args.l1_ratio,
        'class_weight': args.class_weight,
        'max_iter': args.max_iter,
        'solver': resolve_solver(args.l1_ratio),
        'seed': args.seed,
        'test_size': args.test_size,
        'cv_folds': args.cv_folds,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'imputacion_age': 'mediana por sex x pclass (en pipeline)',
        'imputacion_embarked': 'moda (en pipeline)',
        'transformacion_fare': 'log1p',
    }

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name or default_run_name(args)) as run:
        mlflow.set_tags({'tipo_modelo': 'LogisticRegression', 'dataset': 'seaborn-titanic'})

        for name, value in params.items():
            mlflow.log_param(name, value)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        mlflow.log_metric('n_iter', int(pipeline.named_steps['model'].n_iter_[0]))

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            save_confusion_matrix(y_train, y_oof, f'Matriz de confusión (CV, {args.cv_folds} folds)',
                                  tmp / 'confusion_matrix_cv.png')
            save_coefficients(pipeline, tmp / 'coeficientes.csv')
            (tmp / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
            if args.eval_test:
                save_confusion_matrix(y_test, pipeline.predict(X_test), 'Matriz de confusión (test)',
                                      tmp / 'confusion_matrix_test.png')
            for artifact in sorted(tmp.iterdir()):
                mlflow.log_artifact(str(artifact))

        mlflow.sklearn.log_model(
            pipeline,
            name='model',
            signature=infer_signature(X_train, pipeline.predict(X_train)),
            input_example=X_train.head(5),
            # skops (formato por defecto) no serializa transformadores propios como GroupMedianImputer
            serialization_format='cloudpickle',
        )

    return run.info.run_id, metrics


def main(argv=None):
    args = parse_args(argv)
    run_id, metrics = run_experiment(args)
    print(f'Experimento: {args.experiment_name} | Run ID: {run_id}')
    for name, value in metrics.items():
        print(f'  {name}: {value:.4f}')


if __name__ == '__main__':
    main()
