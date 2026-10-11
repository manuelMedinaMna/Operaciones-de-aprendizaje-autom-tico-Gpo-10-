# Código de experimentación base con MLflow

Regresión Logística sobre el dataset Titanic (seaborn), migrada del notebook
`TC5061_Notebook_Baseline_MedinaJuarez.ipynb` a un script parametrizable e
instrumentado con MLflow Tracking.

## Instalación

Probado con Python 3.13.11.

```bash
pip install -r requirements.txt
```

## Ejecución

Los comandos se ejecutan desde esta carpeta, porque el backend (`mlflow.db`) y
los artefactos (`mlruns/`) se resuelven de forma relativa al directorio actual.

```bash
# Valores por defecto (reproduce el baseline del notebook)
python titanic_logistic_regression_pipeline.py

# Variando hiperparámetros
python titanic_logistic_regression_pipeline.py --C 0.1 --l1-ratio 0.5 --class-weight balanced --seed 7

# Interfaz para comparar corridas
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

| Argumento | Default | Descripción |
|---|---|---|
| `--C` | `1.0` | Inverso de la fuerza de regularización. |
| `--l1-ratio` | `0.0` | Mezcla de la penalización: 0 = L2, 1 = L1, intermedio = Elastic-Net. |
| `--class-weight` | `none` | `none` o `balanced`. |
| `--max-iter` | `1000` | Máximo de iteraciones del solver. |
| `--seed` | `42` | Semilla aleatoria. |
| `--test-size` | `0.2` | Proporción del conjunto de prueba. |
| `--cv-folds` | `5` | Folds de la validación cruzada estratificada. |
| `--eval-test` | apagado | Evalúa también en el conjunto de prueba. |
| `--experiment-name` | `titanic-logreg-hiperparametros` | Experimento de MLflow. |
| `--tracking-uri` | `sqlite:///mlflow.db` | Backend de MLflow Tracking. |
| `--run-name` | automático | Nombre de la corrida. |

El solver se elige según `--l1-ratio`: `lbfgs` cuando es 0 y `saga` en cualquier
otro caso, porque `lbfgs` solo admite penalización L2.

## Qué registra cada corrida

- **Parámetros** (`log_param`): hiperparámetros, solver, semilla, configuración de la partición y de la validación cruzada, y decisiones de preprocesamiento.
- **Métricas** (`log_metric`): accuracy, F1 y PR-AUC en validación cruzada (media y desviación) y en train, además del número de iteraciones. Con `--eval-test` se agregan las métricas de prueba.
- **Artefactos** (`log_artifact`): `confusion_matrix_cv.png` (predicciones fuera de fold), `coeficientes.csv` y `metrics.json`.
- **Modelo** (`mlflow.sklearn.log_model`): el pipeline completo, con firma y ejemplo de entrada. Recibe las columnas crudas, porque la imputación y la creación de variables están dentro del pipeline.

Las métricas de comparación provienen de la validación cruzada sobre train. El
conjunto de prueba queda reservado para el modelo final (`--eval-test`), de modo
que no influya en la elección de hiperparámetros.

## Corridas del experimento `titanic-logreg-hiperparametros`

Todas con `--seed 42`, ordenadas por F1 en validación cruzada.

| `C` | `l1_ratio` | `class_weight` | Solver | Accuracy CV | F1 CV | PR-AUC CV |
|---|---|---|---|---|---|---|
| 10 | 0 | none | lbfgs | 0.8188 | 0.7621 | 0.8490 |
| 1 | 0 | none | lbfgs | 0.8188 | 0.7610 | 0.8488 |
| 0.1 | 0 | none | lbfgs | 0.8188 | 0.7596 | 0.8480 |
| 1 | 1 | none | saga | 0.8160 | 0.7583 | 0.8532 |
| 1 | 0 | balanced | lbfgs | 0.8006 | 0.7529 | 0.8456 |
| 0.1 | 0.5 | none | saga | 0.8062 | 0.7402 | 0.8512 |
| 0.01 | 0 | none | lbfgs | 0.7809 | 0.6453 | 0.8354 |

## Reproducibilidad

Una sola semilla (`--seed`) controla todas las fuentes de aleatoriedad:

| Fuente | Dónde se fija |
|---|---|
| Partición train/test | `train_test_split(..., random_state=seed)` |
| Folds de validación cruzada | `StratifiedKFold(shuffle=True, random_state=seed)` |
| Solver del modelo (`saga` es estocástico) | `LogisticRegression(random_state=seed)` |
| Generadores globales | `random.seed(seed)` y `np.random.seed(seed)` |

La semilla se registra como parámetro de cada corrida. Las versiones de las
dependencias están fijadas en `requirements.txt`.

```bash
python titanic_logistic_regression_pipeline.py --C 0.1 --l1-ratio 0.5 --seed 7 --experiment-name titanic-logreg-reproducibilidad
```

Cambiar la semilla cambia también la partición train/test, así que las corridas
con semillas distintas no son comparables entre sí en las métricas de prueba.
