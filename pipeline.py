import copy
import hashlib
import importlib
import json
import os
from datetime import datetime
from datetime import timedelta

import joblib
import pandas as pd
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.python_operator import BranchPythonOperator
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold

import task


def load_task():
    importlib.reload(task)


def load_data_meta(dag_folder, **context):
    data_path = os.path.join(dag_folder, "data", "data.csv")
    ti = context["ti"]
    ti.xcom_push(key="data_md5", value=calc_md5(data_path))
    ti.xcom_push(key="data_path", value=data_path)


def calc_md5(file_path):
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


def load_model_meta(dag_folder, **context):
    model_path = os.path.join(dag_folder, "model", "model.pkl")
    meta_path = os.path.join(dag_folder, "model", "meta.json")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    score = meta.get("score") or 0
    ti = context["ti"]
    ti.xcom_push(key="saved_score", value=score)
    ti.xcom_push(key="model_path", value=model_path)
    ti.xcom_push(key="meta_path", value=meta_path)
    ti.xcom_push(key="model_meta", value=meta)


def check_data(**context):
    ti = context["ti"]
    model_data_md5 = ti.xcom_pull(task_ids="load_model_meta", key="model_meta")
    data_md5 = ti.xcom_pull(task_ids="load_data_meta", key="data_md5")
    if data_md5 != model_data_md5:
        raise ValueError("Data MD5 mismatch")


def run_model_pipeline(**context):
    ti = context["ti"]
    data_path = ti.xcom_pull(task_ids="load_data_meta", key="data_path")
    meta = ti.xcom_pull(task_ids="load_model_meta", key="model_meta")
    meta = copy.deepcopy(meta)
    model_path = ti.xcom_pull(task_ids="load_model_meta", key="model_path")
    meta_path = ti.xcom_pull(task_ids="load_model_meta", key="meta_path")
    data = pd.read_csv(data_path)
    features = data.drop("price", axis=1)
    target = data["price"]

    model = task.model
    scorer = make_scorer(mean_absolute_percentage_error)
    score = cv_catboost(
        model,
        features,
        target,
        cv=5,
        scoring=scorer
    )
    model.fit(features, target)

    healthcheck_data = meta.get("test_input")
    healthcheck_pred = None
    if healthcheck_data:
        healthcheck_pred = get_healthcheck_prediction(model, healthcheck_data)
    model.save_model(model_path)
    meta["score"] = score
    meta["params"] = task.params
    if healthcheck_pred is not None:
        meta["test_output"] = healthcheck_pred
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    return score


def cv_catboost(model, features, target, cv, scoring):
    """
    Кросс-валидация моделей построенных CatBoost.
    Использует специфичную для катбуста валидацию по eval_set
    для улучшения качества модели.

    :param model:
    :param features:
    :param target:
    :param cv:
    :param scoring:
    :return:
    """
    features = features.copy().reset_index(drop=True)
    target = target.copy().reset_index(drop=True)
    kf = KFold(n_splits=cv)
    scores = []
    for train_idx, val_idx in kf.split(features):
        features_train = features.loc[train_idx, :]
        target_train = target[train_idx]
        features_val = features.loc[val_idx, :]
        target_val = target[val_idx]
        model.fit(
            features_train,
            target_train,
            eval_set=(features_val, target_val),
            use_best_model=True,
        )
        scores.append(scoring(model, features_val, target_val))
    return sum(scores) / len(scores)


def get_healthcheck_prediction(model, data):
    data = pd.DataFrame(data, index=[0])
    return int(model.predict(data)[0])


def evaluate_run(**context):
    ti = context["ti"]
    saved_score = ti.xcom_pull(task_ids="load_model_meta", key="saved_score")
    score = ti.xcom_pull(task_ids="run_model_pipeline", key="return_value")
    if score > saved_score:
        return "save_model"
    return "rollback_model"


dag = DAG(
    "train_evaluate_model",
    default_args={
        "owner": "airflow",
        "depends_on_past": False,
        "start_date": datetime(2024, 4, 18),
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(days=1),
    },
    description="A simple DAG to train and evaluate a model",
    schedule_interval=timedelta(days=1),
)
dvc_pull_task = BashOperator(
    task_id="pull_data",
    bash_command="dvc pull",
    cwd=dag.folder,
    dag=dag
)
load_task_task = PythonOperator(
    task_id="load_task",
    python_callable=load_task,
    dag=dag,
)
load_model_meta_task = PythonOperator(
    task_id="load_model_meta",
    python_callable=load_model_meta,
    op_kwargs={"dag_folder": dag.folder},
    dag=dag,
)
load_data_meta_task = PythonOperator(
    task_id="load_data_meta",
    python_callable=load_data_meta,
    op_kwargs={"dag_folder": dag.folder},
    dag=dag,
)
check_data_task = PythonOperator(
    task_id="check_data",
    python_callable=check_data,
    dag=dag,
)
run_model_pipeline_task = PythonOperator(
    task_id="run_model_pipeline",
    python_callable=run_model_pipeline,
    dag=dag,
)
evaluate_run_task = BranchPythonOperator(
    task_id="evaluate_run",
    python_callable=evaluate_run,
    dag=dag,
)
save_model_task = BashOperator(
    task_id="save_model",
    bash_command=(
        "dvc add {{ task_instance.xcom_pull(task_ids='load_model_meta', key='model_path') }} && \\"  # noqa: E501
        "dvc add {{ task_instance.xcom_pull(task_ids='load_model_meta', key='meta_path'') }} && \\"   # noqa: E501
        "dvc push"
    ),
    cwd=dag.folder,
    dag=dag
)
rollback_model_task = BashOperator(
    task_id="rollback_model",
    bash_command="dvc pull --force",
    cwd=dag.folder,
    dag=dag
)
(
    dvc_pull_task >>
    load_task_task >>
    load_model_meta_task >>
    load_data_meta_task >>
    check_data_task >>
    run_model_pipeline_task >>
    evaluate_run_task >>
    [save_model_task, rollback_model_task]
)
