import argparse
import io
import json
import sys
import traceback
from datetime import datetime, timezone

import boto3
import mlflow
from pyspark.sql import SparkSession
from mlflow.tracking import MlflowClient


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


def read_json_s3(s3, bucket: str, key: str):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def write_json_s3(s3, bucket: str, key: str, payload: dict):
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def object_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def copy_object(s3, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str):
    s3.copy_object(
        Bucket=dst_bucket,
        Key=dst_key,
        CopySource={"Bucket": src_bucket, "Key": src_key},
    )


def load_candidate_results(s3, bucket: str, models_prefix: str):
    base = normalize_prefix(models_prefix) + "results/"
    names = ["v1_reg_base", "v1_reg_shallow", "v1_reg_no_ipc_parts"]
    results = []
    for name in names:
        key = f"{base}{name}.json"
        results.append(read_json_s3(s3, bucket, key))
    return results


def read_current_champion_meta_if_exists(s3, prod_bucket: str, prod_prefix: str):
    meta_key = f"{normalize_prefix(prod_prefix)}champion/meta.json"
    if not object_exists(s3, prod_bucket, meta_key):
        return None
    return read_json_s3(s3, prod_bucket, meta_key)


def promote_candidate(s3, src_bucket: str, prod_bucket: str,
                      candidate: dict, prod_prefix: str):
    prod_prefix = normalize_prefix(prod_prefix)

    champion_model_key = f"{prod_prefix}champion/model.cbm"
    champion_meta_key = f"{prod_prefix}champion/meta.json"

    previous_model_key = f"{prod_prefix}previous/model.cbm"
    previous_meta_key = f"{prod_prefix}previous/meta.json"

    if object_exists(s3, prod_bucket, champion_model_key):
        copy_object(s3, prod_bucket, champion_model_key, prod_bucket, previous_model_key)
    if object_exists(s3, prod_bucket, champion_meta_key):
        copy_object(s3, prod_bucket, champion_meta_key, prod_bucket, previous_meta_key)

    copy_object(
        s3=s3,
        src_bucket=src_bucket,
        src_key=candidate["s3_model_key"],
        dst_bucket=prod_bucket,
        dst_key=champion_model_key,
    )

    champion_meta = {
        "candidate_name": candidate["candidate_name"],
        "model_name": candidate["model_name"],
        "model_version": candidate["model_version"],
        "run_id": candidate["run_id"],
        "val_rmse": candidate["val_rmse"],
        "val_mae": candidate["val_mae"],
        "input_key": candidate["input_key"],
        "source_model_key": candidate["s3_model_key"],
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_s3(s3, prod_bucket, champion_meta_key, champion_meta)

    return champion_model_key, champion_meta_key


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--s3-endpoint", default="https://storage.yandexcloud.net")
    parser.add_argument("--ipc-access-key", required=True)
    parser.add_argument("--ipc-secret-key", required=True)

    parser.add_argument("--bucket", required=True, help="Training bucket with candidate results")
    parser.add_argument("--models-prefix", required=True, help="Training models prefix")

    parser.add_argument("--prod-bucket", required=True, help="Prod bucket for champion serving")
    parser.add_argument("--prod-prefix", required=True, help="Prod prefix, e.g. ipc/")

    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--model-name", required=True)
    
    parser.add_argument("--promote-margin", default="0.0")
    

    args, _ = parser.parse_known_args()

    try:
        promote_margin = float(str(args.promote_margin).strip())
    except Exception:
        raise ValueError(f"Invalid --promote-margin value: {args.promote_margin}")
    
    
    spark = None
    try:
        spark = SparkSession.builder.appName("IPCSelectChampion").getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
        
        s3 = make_s3_client(
            endpoint=args.s3_endpoint,
            access_key=args.ipc_access_key,
            secret_key=args.ipc_secret_key,
        )

        mlflow.set_tracking_uri(args.tracking_uri)
        client = MlflowClient()

        candidates = load_candidate_results(
            s3=s3,
            bucket=args.bucket,
            models_prefix=args.models_prefix,
        )

        best_new = min(candidates, key=lambda c: c["val_rmse"])

        current = read_current_champion_meta_if_exists(
            s3=s3,
            prod_bucket=args.prod_bucket,
            prod_prefix=args.prod_prefix,
        )

        promote = False
        reason = None

        if current is None:
            promote = True
            reason = "no_current_champion"
        else:
            current_rmse = float(current["val_rmse"])
            new_rmse = float(best_new["val_rmse"])
            threshold = current_rmse * (1.0 - promote_margin)
            promote = new_rmse < threshold
            reason = f"new_rmse={new_rmse} threshold={threshold}"

        result = {
            "best_new": best_new,
            "current_champion": current,
            "promote": promote,
            "reason": reason,
        }

        if promote:
            if not best_new.get("model_version"):
                raise ValueError(f"Selected model has no model_version: {best_new}")

            client.set_registered_model_alias(
                name=args.model_name,
                alias="airflow",
                version=str(best_new["model_version"]),
            )

            champion_model_key, champion_meta_key = promote_candidate(
                s3=s3,
                src_bucket=args.bucket,
                prod_bucket=args.prod_bucket,
                candidate=best_new,
                prod_prefix=args.prod_prefix,
            )

            result["promoted_model_key"] = champion_model_key
            result["promoted_meta_key"] = champion_meta_key

        print(json.dumps(result, ensure_ascii=False))

    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()
