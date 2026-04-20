import argparse
import io
import re
import sys
import traceback
import zipfile
from datetime import datetime
from typing import List, Dict, Any, Optional

import boto3
import botocore
import pandas as pd
import xml.etree.ElementTree as ET
from dateutil.relativedelta import relativedelta
from pyspark.sql import SparkSession


IPC_CANON_RE = re.compile(
    r"^([A-H])\s*(\d{2})\s*([A-Z])\s*0*(\d{1,4})\s*/\s*(\d{2,6})$"
)


def prev_bulletins(start_code: str, n: int) -> List[str]:
    """
    start_code: 'YYYYMM', например '202603'
    n: сколько месяцев (включая стартовый)
    """
    year = int(start_code[:4])
    month = int(start_code[4:6])

    dt = datetime(year, month, 1)
    codes = []
    for _ in range(n):
        codes.append(f"{dt.year:04d}{dt.month:02d}")
        dt = dt - relativedelta(months=1)
    return codes


def canonicalize_ipc(raw: Optional[str]) -> Optional[str]:
    """
    Привести IPC-строку к виду 'C07D 401/00',
    отрезав хвост с датой версии, если он есть.
    """
    if raw is None:
        return None

    s = raw.upper().strip()
    s = re.sub(r"\s+", " ", s)

    m = re.search(r"\d{8}", s)
    if m:
        s = s[:m.start()].rstrip()

    s = s.replace(" /", "/").replace("/ ", "/")
    s = re.sub(r"\s+", " ", s).strip()

    s_no_space = s.replace(" ", "")
    m = re.match(r"^([A-H])(\d{2})([A-Z])(\d{1,4})/(\d{2,6})$", s_no_space)
    if m:
        sec, cls, subcls, main_group, subgroup = m.groups()
        return f"{sec}{cls}{subcls} {int(main_group)}/{subgroup}"

    m = IPC_CANON_RE.match(s)
    if m:
        sec, cls, subcls, main_group, subgroup = m.groups()
        return f"{sec}{cls}{subcls} {int(main_group)}/{subgroup}"

    return s or None


def parse_ai_ipc_line(line: str) -> List[Dict[str, Any]]:
    """
    Разобрать строку вида:
    AI_IPC:A61K 31/00 (22.23%);C07D401/00 (8.37%);...
    """
    if not line or "AI_IPC:" not in line:
        return []

    payload = line.split("AI_IPC:", 1)[1].strip()
    if payload.endswith(";"):
        payload = payload[:-1]

    result = []
    for item in payload.split(";"):
        item = item.strip()
        if not item:
            continue

        m = re.match(r"(.+?)\s*\((\d+(?:\.\d+)?)%\)$", item)
        if m:
            ipc_code_raw = m.group(1).strip()
            ipc_code_canon = canonicalize_ipc(ipc_code_raw)
            score = float(m.group(2))
            result.append({
                "ipc_code_raw": ipc_code_raw,
                "ipc_code_canon": ipc_code_canon,
                "score_percent": score,
            })
        else:
            ipc_code_raw = item.strip()
            ipc_code_canon = canonicalize_ipc(ipc_code_raw)
            result.append({
                "ipc_code_raw": ipc_code_raw,
                "ipc_code_canon": ipc_code_canon,
                "score_percent": None,
            })
    return result


def make_s3_client(endpoint: str, access_key: str, secret_key: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def normalize_prefix(prefix: str) -> str:
    prefix = (prefix or "").strip().lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return prefix


def list_available_bulletins(
    s3,
    bucket: str,
    pub_prefix: str,
) -> List[str]:
    """
    Список доступных YYYYMM по ключам pub/YYYYMM.zip в бакете.
    """
    prefix = normalize_prefix(pub_prefix)
    paginator = s3.get_paginator("list_objects_v2")

    codes = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if not key.startswith(prefix) or not key.endswith(".zip"):
                continue

            name = key[len(prefix):]  # например, '202603.zip'
            if len(name) != 10:
                continue

            code = name[:6]
            if code.isdigit():
                codes.add(code)

    return sorted(codes)


def resolve_training_last_month(
    s3,
    bucket: str,
    pub_prefix: str,
    training_last_month: str,
) -> str:
    """
    Если training_last_month == 'last' → берём последний доступный YYYYMM.
    Иначе проверяем, что указанный YYYYMM есть среди доступных.
    """
    available = list_available_bulletins(s3, bucket, pub_prefix)
    if not available:
        raise ValueError("No bulletin ZIPs found in S3 under given prefix")

    if training_last_month == "last":
        return available[-1]

    if not re.match(r"^\d{6}$", training_last_month):
        raise ValueError(
            "training_last_month must be either 'last' or explicit YYYYMM"
        )

    if training_last_month not in available:
        raise ValueError(
            f"Requested training_last_month={training_last_month} not found. "
            f"Available range: {available[0]}..{available[-1]}"
        )

    return training_last_month


def read_s3_text_file(s3, bucket: str, key: str, encoding: str = "utf-8") -> Optional[str]:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read().decode(encoding, errors="ignore")
    except botocore.exceptions.ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise


def read_ai_ipc_from_lst_s3(s3, bucket: str, lst_key: str) -> List[Dict[str, Any]]:
    """
    Найти строку AI_IPC: в .LST из S3 и вернуть распарсенный список.
    """
    txt = read_s3_text_file(s3, bucket, lst_key)
    if not txt:
        return []

    for line in txt.splitlines():
        if line.startswith("AI_IPC:"):
            return parse_ai_ipc_line(line.strip())

    return []


def extract_ipc_from_bulletin_s3(
    s3,
    bucket: str,
    pub_prefix: str,
    lst_prefix: str,
    bulletin_code: str,
) -> pd.DataFrame:
    """
    Читает zip-бюллетень вида pub/YYYYMM.zip из S3 и сопоставляет
    XML внутри него с lst/YYYY....LST из того же бакета.
    """
    zip_key = f"{normalize_prefix(pub_prefix)}{bulletin_code}.zip"
    print(f"Processing s3://{bucket}/{zip_key}")

    try:
        obj = s3.get_object(Bucket=bucket, Key=zip_key)
    except botocore.exceptions.ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            print(f"Skip {bulletin_code}: zip not found")
            return pd.DataFrame()
        raise

    zip_bytes = obj["Body"].read()
    rows = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        xml_files = [name for name in zf.namelist() if name.endswith("Document.xml")]
        print(f"Found {len(xml_files)} xml files in {bulletin_code}")

        for xml_name in xml_files:
            try:
                with zf.open(xml_name) as f:
                    tree = ET.parse(f)
                    root = tree.getroot()

                patent_number = root.findtext(".//B110")
                application_number = root.findtext(".//B210")
                bulletin_date = root.findtext(".//B480/date")

                ipc_nodes = root.findall(".//classification-ipcr/text")
                expert_ipc_raw = []
                expert_ipc_canon = []

                for node in ipc_nodes:
                    if not node.text:
                        continue
                    raw_value = node.text.rstrip()
                    canon_value = canonicalize_ipc(raw_value)
                    expert_ipc_raw.append(raw_value)
                    if canon_value:
                        expert_ipc_canon.append(canon_value)

                expert_ipc_canon = sorted(set(expert_ipc_canon))

                app_num = str(application_number or "").strip()
                if not app_num:
                    rows.append({
                        "source_file": xml_name,
                        "patent_number": patent_number,
                        "application_number": application_number,
                        "bulletin_date": bulletin_date,
                        "expert_ipc_count": len(expert_ipc_canon),
                        "expert_ipc_canon": expert_ipc_canon,
                        "lst_key": None,
                        "lst_found": False,
                        "ai_ipc_count": 0,
                        "ai_ipc_raw": [],
                        "ai_ipc_canon": [],
                        "ai_scores": [],
                        "bulletin_code": bulletin_code,
                        "error": "application_number is empty",
                    })
                    continue

                lst_key = f"{normalize_prefix(lst_prefix)}{app_num}.LST"
                ai_items = read_ai_ipc_from_lst_s3(s3, bucket, lst_key)

                if not ai_items:
                    continue

                ai_ipc_raw = [x["ipc_code_raw"] for x in ai_items]
                ai_ipc_canon = [x["ipc_code_canon"] for x in ai_items]
                ai_scores = [x["score_percent"] for x in ai_items]

                rows.append({
                    "source_file": xml_name,
                    "patent_number": patent_number,
                    "application_number": application_number,
                    "bulletin_date": bulletin_date,
                    "expert_ipc_count": len(expert_ipc_canon),
                    "expert_ipc_canon": expert_ipc_canon,
                    "lst_key": lst_key,
                    "lst_found": True,
                    "ai_ipc_count": len(ai_ipc_canon),
                    "ai_ipc_raw": ai_ipc_raw,
                    "ai_ipc_canon": ai_ipc_canon,
                    "ai_scores": ai_scores,
                    "bulletin_code": bulletin_code,
                    "error": None,
                })

            except Exception as e:
                rows.append({
                    "source_file": xml_name,
                    "patent_number": None,
                    "application_number": None,
                    "bulletin_date": None,
                    "expert_ipc_count": 0,
                    "expert_ipc_canon": [],
                    "lst_key": None,
                    "lst_found": False,
                    "ai_ipc_count": 0,
                    "ai_ipc_raw": [],
                    "ai_ipc_canon": [],
                    "ai_scores": [],
                    "bulletin_code": bulletin_code,
                    "error": str(e),
                })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=[
        "source_file",
        "patent_number",
        "application_number",
        "bulletin_date",
        "expert_ipc_count",
        "expert_ipc_canon",
        "lst_key",
        "lst_found",
        "ai_ipc_count",
        "ai_ipc_raw",
        "ai_ipc_canon",
        "ai_scores",
        "bulletin_code",
        "error",
    ])


def serialize_lists_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Чтобы не зависеть от сложных list/array типов при дальнейшем чтении,
    сохраняем списки как pipe-separated строки.
    """
    out = df.copy()

    out["expert_ipc_canon"] = out["expert_ipc_canon"].apply(
        lambda x: "|".join(x) if isinstance(x, list) else ""
    )
    out["ai_ipc_raw"] = out["ai_ipc_raw"].apply(
        lambda x: "|".join(x) if isinstance(x, list) else ""
    )
    out["ai_ipc_canon"] = out["ai_ipc_canon"].apply(
        lambda x: "|".join(x) if isinstance(x, list) else ""
    )
    out["ai_scores"] = out["ai_scores"].apply(
        lambda x: "|".join("" if v is None else str(v) for v in x)
        if isinstance(x, list) else ""
    )

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--s3-endpoint", default="https://storage.yandexcloud.net")
    parser.add_argument("--ipc-access-key", required=True)
    parser.add_argument("--ipc-secret-key", required=True)

    parser.add_argument("--bucket", required=True, help="Bucket with source and target data, e.g. ipc")
    parser.add_argument("--pub-prefix", default="pub/", help="Prefix for monthly ZIP bulletins")
    parser.add_argument("--lst-prefix", default="lst/", help="Prefix for .LST files")
    parser.add_argument("--output-prefix", required=True, help="Output prefix in the same bucket, e.g. dataprep/")

    parser.add_argument(
        "--training-last-month",
        required=True,
        help="Either 'last' or explicit bulletin code YYYYMM"
    )
    parser.add_argument(
        "--months",
        type=int,
        required=True,
        help="How many months to process, including last month"
    )

    args, _ = parser.parse_known_args()

    print("ARGS:", sys.argv)

    spark = SparkSession.builder.appName("IPCPrep").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

    s3 = make_s3_client(
        endpoint=args.s3_endpoint,
        access_key=args.ipc_access_key,
        secret_key=args.ipc_secret_key,
    )

    try:
        resolved_last_month = resolve_training_last_month(
            s3=s3,
            bucket=args.bucket,
            pub_prefix=args.pub_prefix,
            training_last_month=args.training_last_month,
        )

        print(f"Resolved training_last_month: {resolved_last_month}")

        bulletin_codes = prev_bulletins(resolved_last_month, args.months)
        print(f"Bulletin window: {bulletin_codes}")

        all_dfs = []

        for code in bulletin_codes:
            df_month = extract_ipc_from_bulletin_s3(
                s3=s3,
                bucket=args.bucket,
                pub_prefix=args.pub_prefix,
                lst_prefix=args.lst_prefix,
                bulletin_code=code,
            )

            if df_month is None or df_month.empty:
                print(f"No usable data for bulletin {code}")
                continue

            print(f"Collected {len(df_month)} rows for bulletin {code}")
            all_dfs.append(df_month)

        if not all_dfs:
            print("No data collected")
            return

        df_all = pd.concat(all_dfs, ignore_index=True)
        print("rows total:", len(df_all))

        df_to_save = serialize_lists_for_parquet(df_all)

        output_prefix = normalize_prefix(args.output_prefix)
        output_key = (
            f"{output_prefix}"
            f"ipc_with_ai_{resolved_last_month}_last_{args.months}.parquet"
        )

        print(f"Writing parquet to s3://{args.bucket}/{output_key}")

        buf = io.BytesIO()
        df_to_save.to_parquet(buf, index=False, engine="pyarrow")
        buf.seek(0)

        s3.put_object(
            Bucket=args.bucket,
            Key=output_key,
            Body=buf.getvalue(),
            ContentType="application/octet-stream",
        )

        print("Done")
        print(f"Saved rows: {len(df_to_save)}")
        print(f"Saved object: s3://{args.bucket}/{output_key}")

    except Exception as e:
        print("ERROR:", str(e), file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()