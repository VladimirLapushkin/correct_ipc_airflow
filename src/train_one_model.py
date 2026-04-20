import argparse
import ast
import re
import io
import json
import os
import sys
import tempfile
import traceback
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional

import boto3
import pandas as pd
import mlflow
import mlflow.catboost
from mlflow.tracking import MlflowClient

from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error
from pyspark.sql import SparkSession


@dataclass
class TrainConfig:
    input_key: str
    models_prefix: str
    target_col: str = "y"
    random_state: int = 42
    train_frac: float = 2 / 3
    top_k: int = 10


def make_s3_client(endpoint: str, access_key: str, secret_key: str):
    return boto3.client(
        "s3",
        region_name="ru-central1",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def normalize_prefix(prefix: str) -> str:
    prefix = (prefix or "").strip().lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return prefix


def list_available_datasets(
    s3,
    bucket: str,
    output_prefix: str,
) -> List[str]:
    """
    Ищем файлы вида:
      <output_prefix>ipc_with_ai_YYYYMM_last_N.parquet
    и возвращаем список кодов YYYYMM, отсортированный.
    """
    prefix = normalize_prefix(output_prefix)
    paginator = s3.get_paginator("list_objects_v2")

    codes = set()
    pattern = re.compile(r"^ipc_with_ai_(\d{6})_last_(\d+)\.parquet$")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.startswith(prefix):
                continue

            name = key[len(prefix):]
            m = pattern.match(name)
            if not m:
                continue

            code = m.group(1)
            codes.add(code)

    return sorted(codes)


def resolve_training_last_month_for_dataset(
    s3,
    bucket: str,
    output_prefix: str,
    training_last_month: str,
) -> str:
   

    """
    Если training_last_month == 'last' → берём последний доступный YYYYMM
    среди parquet-файлов.
    Иначе проверяем, что указанный YYYYMM есть.
    """
    available = list_available_datasets(s3, bucket, output_prefix)
    if not available:
        raise ValueError("No parquet datasets found under given output-prefix")

    if training_last_month == "last":
        return available[-1]

    if not re.match(r"^\d{6}$", training_last_month):
        raise ValueError(
            "training_last_month must be either 'last' or explicit YYYYMM"
        )

    if training_last_month not in available:
        raise ValueError(
            f"Requested training_last_month={training_last_month} not found among "
            f"prepared datasets. Available range: {available[0]}..{available[-1]}"
        )

    return training_last_month


def build_input_key(
    resolved_last_month: str,
    months: int,
    output_prefix: str,
) -> str:
    """
    !B@>8< 8<O D09;0 2 B>< 65 D>@<0B5, GB> data_prep.py:
      ipc_with_ai_<resolved_last_month>_last_<months>.parquet
    """
    prefix = normalize_prefix(output_prefix)
    return f"{prefix}ipc_with_ai_{resolved_last_month}_last_{months}.parquet"


def read_parquet_from_s3(
    s3,
    bucket: str,
    key: str,
) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def to_main_group(code: str) -> Optional[str]:
    if not code or not isinstance(code, str):
        return None
    try:
        main_part, _ = code.split("/")
        return f"{main_part.strip()}/00"
    except Exception:
        return None


def parse_list_field(val) -> List[str]:
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                return []
        return [x for x in s.split("|") if x]
    return []


def parse_ipc(code: str) -> Dict[str, Any]:
    if not code or not isinstance(code, str):
        return dict(section=None, class2=None, subclass=None, main_group=None, subgroup=None)
    try:
        main_part, subpart = code.split("/")
        main_part = main_part.strip()
        subpart = subpart.strip()

        section = main_part[0]
        class2 = main_part[1:3]
        subclass = main_part[3]
        main_group = int(main_part[4:].strip())
        subgroup = int(subpart)

        return dict(
            section=section,
            class2=class2,
            subclass=subclass,
            main_group=main_group,
            subgroup=subgroup,
        )
    except Exception:
        return dict(section=None, class2=None, subclass=None, main_group=None, subgroup=None)


def build_long_dataset(df_wide: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for rec in df_wide.to_dict(orient="records"):
        expert_list = parse_list_field(rec.get("expert_ipc_canon") or [])
        ai_list = parse_list_field(rec.get("ai_ipc_canon") or [])
        ai_scores = rec.get("ai_scores") or []

        if isinstance(ai_scores, str):
            s = ai_scores.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = ast.literal_eval(s)
                    ai_scores = [float(x) for x in parsed] if isinstance(parsed, list) else []
                except Exception:
                    ai_scores = []
            else:
                ai_scores = [float(x) for x in s.split("|")] if s else []

        expert_exact = set(expert_list)
        expert_main = set(
            to_main_group(c) for c in expert_list if to_main_group(c) is not None
        )

        patent_id = rec.get("application_number")
        bulletin_code = rec.get("bulletin_code")

        for idx, ipc in enumerate(ai_list):
            score = ai_scores[idx] if idx < len(ai_scores) else None
            ipc_exact = ipc
            ipc_main = to_main_group(ipc)

            if ipc_exact in expert_exact:
                y = 1.0
            elif ipc_main is not None and ipc_main in expert_main:
                y = 0.5
            else:
                y = 0.0

            row = {
                "patent_id": patent_id,
                "bulletin_code": bulletin_code,
                "ipc_code": ipc,
                "ipc_main": ipc_main,
                "ai_score": score,
                "rank": idx + 1,
                "y": y,
            }
            row.update(parse_ipc(ipc))
            rows.append(row)

    return pd.DataFrame(rows)


def split_train_val_by_time(df_long: pd.DataFrame, train_frac: float = 2.0 / 3.0):
    months = sorted(df_long["bulletin_code"].dropna().unique())
    n_months = len(months)
    split_idx = int(n_months * train_frac)

    train_months = months[:split_idx]
    val_months = months[split_idx:]

    df_train = df_long[df_long["bulletin_code"].isin(train_months)].copy()
    df_val = df_long[df_long["bulletin_code"].isin(val_months)].copy()
    return df_train, df_val


def train_model_v1_reg(df_train: pd.DataFrame,
                       df_val: pd.DataFrame,
                       base_cfg: TrainConfig,
                       model_cfg: Dict[str, Any]):
    use_ipc_parts = model_cfg.get("use_ipc_parts", True)

    feature_cols = ["ai_score", "rank"]
    cat_features = []

    if use_ipc_parts:
        feature_cols += ["main_group", "subgroup", "section", "class2", "subclass"]
        cat_features += ["section", "class2", "subclass"]

    for c in feature_cols:
        if df_train[c].dtype.kind in "biufc":
            df_train[c] = df_train[c].fillna(0)
            df_val[c] = df_val[c].fillna(0)
        else:
            df_train[c] = df_train[c].fillna("NA")
            df_val[c] = df_val[c].fillna("NA")

    X_train = df_train[feature_cols]
    y_train = df_train[base_cfg.target_col]
    X_val = df_val[feature_cols]
    y_val = df_val[base_cfg.target_col]

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    model = CatBoostRegressor(
        depth=model_cfg.get("depth", 6),
        learning_rate=model_cfg.get("learning_rate", 0.05),
        loss_function="RMSE",
        eval_metric="RMSE",
        random_state=base_cfg.random_state,
        verbose=False,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_pred = model.predict(val_pool)
    diff = y_val.values - val_pred
    rmse = float((diff ** 2).mean() ** 0.5)
    mae = mean_absolute_error(y_val, val_pred)

    metrics = {
        "val_rmse": float(rmse),
        "val_mae": float(mae),
        "n_train": int(len(df_train)),
        "n_val": int(len(df_val)),
    }
    return model, metrics


def get_candidate_config(candidate_name: str) -> Dict[str, Any]:
    configs = {
        "v1_reg_base": {"name": "v1_reg_base", "depth": 6, "learning_rate": 0.05, "use_ipc_parts": True},
        "v1_reg_shallow": {"name": "v1_reg_shallow", "depth": 4, "learning_rate": 0.05, "use_ipc_parts": True},
        "v1_reg_no_ipc_parts": {"name": "v1_reg_no_ipc_parts", "depth": 6, "learning_rate": 0.05, "use_ipc_parts": False},
    }
    if candidate_name not in configs:
        raise ValueError(f"Unknown candidate_name: {candidate_name}")
    return configs[candidate_name]


def save_candidate_artifacts_to_s3(s3, bucket: str, models_prefix: str, candidate_name: str,
                                   model, metrics: Dict[str, Any], cfg: TrainConfig):
    models_prefix = normalize_prefix(models_prefix)

    model_key = f"{models_prefix}candidates/{candidate_name}.cbm"
    meta_key = f"{models_prefix}candidates/{candidate_name}_meta.json"

    with tempfile.TemporaryDirectory() as tmpdir:
        local_model_path = os.path.join(tmpdir, f"{candidate_name}.cbm")
        model.save_model(local_model_path)
        with open(local_model_path, "rb") as f:
            model_bytes = f.read()

    s3.put_object(
        Bucket=bucket,
        Key=model_key,
        Body=model_bytes,
        ContentType="application/octet-stream",
    )

    meta = {
        "candidate_name": candidate_name,
        "metrics": metrics,
        "config": asdict(cfg),
        "model_key": model_key,
    }
    s3.put_object(
        Bucket=bucket,
        Key=meta_key,
        Body=json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    return model_key, meta_key


def write_result_json(s3, bucket: str, models_prefix: str, candidate_name: str, result: Dict[str, Any]):
    result_key = f"{normalize_prefix(models_prefix)}results/{candidate_name}.json"
    s3.put_object(
        Bucket=bucket,
        Key=result_key,
        Body=json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    return result_key


def find_model_version_by_run_id_with_retry(
    client: MlflowClient,
    model_name: str,
    run_id: str,
    max_attempts: int = 10,
    sleep_seconds: int = 3,
) -> Optional[str]:
    for attempt in range(1, max_attempts + 1):
        versions = client.search_model_versions(f"name = '{model_name}'")
        for mv in versions:
            if mv.run_id == run_id:
                return str(mv.version)

        print(
            f"Model version not found yet for run_id={run_id}, "
            f"model_name={model_name}, attempt={attempt}/{max_attempts}"
        )
        time.sleep(sleep_seconds)

    return None


def find_model_version_by_run_id(client: MlflowClient, model_name: str, run_id: str) -> Optional[str]:
    versions = client.search_model_versions(f"name = '{model_name}'")
    for mv in versions:
        if mv.run_id == run_id:
            return str(mv.version)
    return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--s3-endpoint", default="https://storage.yandexcloud.net")
    parser.add_argument("--ipc-access-key", required=True)
    parser.add_argument("--ipc-secret-key", required=True)

    parser.add_argument("--bucket", required=True)
    parser.add_argument("--output-prefix", required=True)  # dataprep/
    parser.add_argument("--models-prefix", required=True)

    parser.add_argument(
        "--training-last-month",
        required=True,
        help="Either 'last' or explicit YYYYMM, matching prepared parquet",
    )
    parser.add_argument(
        "--months",
        type=int,
        required=True,
        help="How many months were used in dataprep (N in ipc_with_ai_YYYYMM_last_N.parquet)",
    )

    parser.add_argument("--candidate-name", required=True)

    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--auto-register", action="store_true")

    args, _ = parser.parse_known_args()
    print("ARGS:", sys.argv)

    os.environ["AWS_ACCESS_KEY_ID"] = args.ipc_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = args.ipc_secret_key
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = args.s3_endpoint

    spark = None
    try:
        spark = SparkSession.builder.appName(f"IPCTrain_{args.candidate_name}").getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment_name)
        mlflow_client = MlflowClient()

        s3 = make_s3_client(
            endpoint=args.s3_endpoint,
            access_key=args.ipc_access_key,
            secret_key=args.ipc_secret_key,
        )

        
        resolved_last_month = resolve_training_last_month_for_dataset(
            s3=s3,
            bucket=args.bucket,
            output_prefix=args.output_prefix,
            training_last_month=args.training_last_month,
        )
        print(f"Resolved training_last_month for training: {resolved_last_month}")

        input_key = build_input_key(
            resolved_last_month=resolved_last_month,
            months=args.months,
            output_prefix=args.output_prefix,
        )
        print(f"Using dataset: s3://{args.bucket}/{input_key}")

        df_wide = read_parquet_from_s3(
            s3=s3,
            bucket=args.bucket,
            key=input_key,
        )

        cfg = TrainConfig(
            input_key=input_key,
            models_prefix=args.models_prefix,
        )

        df_long = build_long_dataset(df_wide)
        df_train, df_val = split_train_val_by_time(df_long, train_frac=cfg.train_frac)

        mc = get_candidate_config(args.candidate_name)

        with mlflow.start_run(run_name=args.candidate_name) as run:
            mlflow.log_param("candidate_name", args.candidate_name)
            mlflow.log_param("input_key", cfg.input_key)
            mlflow.log_param("train_frac", cfg.train_frac)
            mlflow.log_param("random_state", cfg.random_state)
            mlflow.log_param("training_last_month_resolved", resolved_last_month)
            mlflow.log_param("months_window", args.months)

            for k, v in mc.items():
                mlflow.log_param(k, v)

            model, metrics = train_model_v1_reg(df_train, df_val, cfg, mc)
            mlflow.log_metrics(metrics)

            model_key, meta_key = save_candidate_artifacts_to_s3(
                s3=s3,
                bucket=args.bucket,
                models_prefix=args.models_prefix,
                candidate_name=args.candidate_name,
                model=model,
                metrics=metrics,
                cfg=cfg,
            )

            mlflow.log_param("s3_model_key", model_key)
            mlflow.log_param("s3_meta_key", meta_key)

            model_version = None
            if args.auto_register:
                mlflow.catboost.log_model(
                    cb_model=model,
                    artifact_path="model",
                    registered_model_name=args.model_name,
                )
            model_version = find_model_version_by_run_id_with_retry(
                client=mlflow_client,
                model_name=args.model_name,
                run_id=run.info.run_id,
                max_attempts=10,
                sleep_seconds=3,
            )
            if model_version is None:
                raise RuntimeError(
                    f"Could not resolve model_version for model_name={args.model_name}, "
                    f"run_id={run.info.run_id} after retry"
                )

            result = {
                "candidate_name": args.candidate_name,
                "run_id": run.info.run_id,
                "model_name": args.model_name,
                "model_version": model_version,
                "val_rmse": metrics["val_rmse"],
                "val_mae": metrics["val_mae"],
                "s3_model_key": model_key,
                "s3_meta_key": meta_key,
                "input_key": input_key,
                "training_last_month_resolved": resolved_last_month,
                "months_window": args.months,
            }

            result_key = write_result_json(
                s3=s3,
                bucket=args.bucket,
                models_prefix=args.models_prefix,
                candidate_name=args.candidate_name,
                result=result,
            )

            print(json.dumps({"result_key": result_key, **result}, ensure_ascii=False))

    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()
