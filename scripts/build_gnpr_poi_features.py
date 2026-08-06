#!/usr/bin/env python3
"""Build full sparse GNPR-SID POI features with Spark."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLUS_CODE_ALPHABET = "23456789CFGHJMPQRVWX"
SCHEMA_VERSION = "gnpr-poi-features-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从北京 POI 和 003 history-10 订单构建 GNPR-SID 的类别、"
            "Plus Code6、Top10 访问小时和 Top10 访问用户稀疏特征。"
        )
    )
    parser.add_argument("--poi-dir", type=Path, required=True, help="POI JSONL 目录。")
    parser.add_argument(
        "--orders-dir",
        type=Path,
        required=True,
        help="003 history-10 订单 JSONL 目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="不存在的正式输出目录。失败时保留同名 .building 目录。",
    )
    parser.add_argument("--train-start", default="20260701", help="训练开始日期。")
    parser.add_argument("--train-end", default="20260712", help="训练结束日期。")
    parser.add_argument("--top-k", type=int, default=10, help="小时和用户 TopK。")
    parser.add_argument(
        "--plus-code-length",
        type=int,
        default=6,
        help="GNPR 区域 Plus Code 长度，当前固定为 6。",
    )
    parser.add_argument(
        "--output-partitions",
        type=int,
        default=32,
        help="POI 特征和用户词表输出分片数。",
    )
    parser.add_argument(
        "--shuffle-partitions",
        type=int,
        default=512,
        help="Spark SQL shuffle 分区数。",
    )
    parser.add_argument(
        "--expected-poi-count",
        type=int,
        default=None,
        help="可选的全量 POI 行数门禁。",
    )
    parser.add_argument(
        "--expected-order-count",
        type=int,
        default=None,
        help="可选的全量订单行数门禁。",
    )
    parser.add_argument(
        "--expected-train-target-count",
        type=int,
        default=None,
        help="可选的训练时间窗目标订单行数门禁。",
    )
    parser.add_argument(
        "--max-pois",
        type=int,
        default=None,
        help="仅用于 smoke 的 POI 行数上限。",
    )
    parser.add_argument(
        "--max-orders",
        type=int,
        default=None,
        help="仅用于 smoke 的订单行数上限。",
    )
    return parser.parse_args()


def _local_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _spark_path(path: Path) -> str:
    return _local_path(path).resolve().as_uri()


def _validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    poi_dir = _local_path(args.poi_dir).resolve()
    orders_dir = _local_path(args.orders_dir).resolve()
    output_dir = _local_path(args.output_dir).resolve()
    staging_dir = output_dir.with_name(output_dir.name + ".building")
    if not poi_dir.is_dir():
        raise ValueError(f"POI 目录不存在：{poi_dir}")
    if not orders_dir.is_dir():
        raise ValueError(f"订单目录不存在：{orders_dir}")
    if output_dir.exists():
        raise ValueError(f"输出目录已存在：{output_dir}")
    if staging_dir.exists():
        raise ValueError(f"构建临时目录已存在：{staging_dir}")
    for name, value in (("train-start", args.train_start), ("train-end", args.train_end)):
        try:
            datetime.strptime(value, "%Y%m%d")
        except ValueError as error:
            raise ValueError(f"{name} 必须是 YYYYMMDD：{value}") from error
    if args.train_start > args.train_end:
        raise ValueError("train-start 不能晚于 train-end")
    if args.top_k <= 0:
        raise ValueError("top-k 必须大于 0")
    if args.plus_code_length != 6:
        raise ValueError("当前 GNPR baseline 的 plus-code-length 固定为 6")
    if args.output_partitions <= 0 or args.shuffle_partitions <= 0:
        raise ValueError("Spark 分区数必须大于 0")
    if args.max_pois is not None and args.max_pois <= 0:
        raise ValueError("max-pois 必须大于 0")
    if args.max_orders is not None and args.max_orders <= 0:
        raise ValueError("max-orders 必须大于 0")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return poi_dir, orders_dir, output_dir, staging_dir


def _encode_plus_code6(longitude: float, latitude: float) -> str | None:
    if longitude is None or latitude is None:
        return None
    try:
        lon = float(longitude)
        lat = float(latitude)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(lon) or not math.isfinite(lat):
        return None
    if not -180.0 <= lon <= 180.0 or not -90.0 <= lat <= 90.0:
        return None
    if lat == 90.0:
        lat -= 0.05
    lon = ((lon + 180.0) % 360.0) - 180.0
    latitude_value = lat + 90.0
    longitude_value = lon + 180.0
    place_value = 20.0
    characters = []
    for _ in range(3):
        latitude_digit = int(latitude_value / place_value)
        longitude_digit = int(longitude_value / place_value)
        characters.append(PLUS_CODE_ALPHABET[latitude_digit])
        characters.append(PLUS_CODE_ALPHABET[longitude_digit])
        latitude_value -= latitude_digit * place_value
        longitude_value -= longitude_digit * place_value
        place_value /= 20.0
    return "".join(characters) + "00+"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_indexed_vocab(dataframe, value_column: str, index_column: str, partitions: int):
    from pyspark.sql import types as T

    ordered = (
        dataframe.select(value_column)
        .distinct()
        .repartitionByRange(partitions, value_column)
        .sortWithinPartitions(value_column)
    )
    indexed = ordered.rdd.map(lambda row: row[0]).zipWithIndex().map(
        lambda item: (item[0], int(item[1]))
    )
    return dataframe.sql_ctx.createDataFrame(
        indexed,
        T.StructType(
            [
                T.StructField(value_column, T.StringType(), False),
                T.StructField(index_column, T.LongType(), False),
            ]
        ),
    )


def _timestamp(column, functions):
    return functions.to_timestamp(
        functions.regexp_replace(functions.substring(column, 1, 19), "T", " ")
    )


def _run(args: argparse.Namespace) -> dict:
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    from pyspark import StorageLevel
    from pyspark.sql import SparkSession, functions as F, types as T
    from pyspark.sql.window import Window

    poi_dir, orders_dir, output_dir, staging_dir = _validate_args(args)
    staging_dir.mkdir()
    spark = (
        SparkSession.builder.appName("poi-gr-gnpr-features-history10")
        .config("spark.pyspark.python", os.environ["PYSPARK_PYTHON"])
        .config("spark.pyspark.driver.python", os.environ["PYSPARK_DRIVER_PYTHON"])
        .config("spark.sql.session.timeZone", "Asia/Shanghai")
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(
        "[GNPR] python_driver={}, python_worker={}".format(
            sys.executable, spark.sparkContext.pythonExec
        ),
        flush=True,
    )

    poi_schema = T.StructType(
        [
            T.StructField("poi_id", T.StringType(), True),
            T.StructField("category_code", T.StringType(), True),
            T.StructField("lng", T.DoubleType(), True),
            T.StructField("lat", T.DoubleType(), True),
        ]
    )
    history_schema = T.ArrayType(
        T.StructType(
            [
                T.StructField("event_time", T.StringType(), True),
                T.StructField("order_id", T.StringType(), True),
                T.StructField("searchid", T.StringType(), True),
                T.StructField("poi_id", T.StringType(), True),
            ]
        )
    )
    order_schema = T.StructType(
        [
            T.StructField("order_id", T.StringType(), True),
            T.StructField("searchid", T.StringType(), True),
            T.StructField("passenger_id", T.StringType(), True),
            T.StructField("poi_id", T.StringType(), True),
            T.StructField("birth_time", T.StringType(), True),
            T.StructField("create_time", T.StringType(), True),
            T.StructField("source_dt", T.StringType(), True),
            T.StructField("history_length", T.IntegerType(), True),
            T.StructField("history_sequence", history_schema, True),
        ]
    )

    try:
        raw_pois = spark.read.schema(poi_schema).json(_spark_path(poi_dir))
        raw_orders = spark.read.schema(order_schema).json(_spark_path(orders_dir))
        if args.max_pois is not None:
            raw_pois = raw_pois.limit(args.max_pois)
        if args.max_orders is not None:
            raw_orders = raw_orders.limit(args.max_orders)

        orders = (
            raw_orders.select(
                F.trim(F.col("order_id")).alias("order_id"),
                F.trim(F.col("searchid")).alias("searchid"),
                F.trim(F.col("passenger_id")).alias("passenger_id"),
                F.trim(F.col("poi_id")).alias("poi_id"),
                "birth_time",
                "create_time",
                F.trim(F.col("source_dt")).alias("source_dt"),
                "history_length",
                "history_sequence",
            )
            .persist(StorageLevel.DISK_ONLY)
        )
        order_summary = orders.agg(
            F.count("*").alias("order_count"),
            F.max(F.coalesce("history_length", F.lit(0))).alias("max_history_length"),
        ).first()
        order_count = int(order_summary["order_count"])
        max_history_length = int(order_summary["max_history_length"] or 0)
        if args.expected_order_count is not None and order_count != args.expected_order_count:
            raise ValueError(
                f"订单行数不一致：{order_count} != {args.expected_order_count}"
            )
        if max_history_length > 10:
            raise ValueError(f"history_length 最大值超过 10：{max_history_length}")
        print(
            f"[GNPR] orders={order_count:,}, max_history_length={max_history_length}",
            flush=True,
        )

        plus_code_udf = F.udf(_encode_plus_code6, T.StringType())
        poi_selected = raw_pois.select(
            F.trim(F.col("poi_id")).alias("poi_id"),
            F.trim(F.col("category_code")).alias("category_code"),
            F.col("lng").cast("double").alias("lng"),
            F.col("lat").cast("double").alias("lat"),
        )
        invalid_poi_count = poi_selected.filter(
            F.col("poi_id").isNull()
            | (F.length("poi_id") == 0)
            | F.col("category_code").isNull()
            | (F.length("category_code") == 0)
            | F.col("lng").isNull()
            | F.col("lat").isNull()
            | F.isnan("lng")
            | F.isnan("lat")
            | (F.col("lng") < -180.0)
            | (F.col("lng") > 180.0)
            | (F.col("lat") < -90.0)
            | (F.col("lat") > 90.0)
        ).count()
        if invalid_poi_count:
            raise ValueError(f"POI 静态字段无效行数：{invalid_poi_count}")
        poi_static = (
            poi_selected.withColumn(
                "plus_code_region",
                plus_code_udf("lng", "lat"),
            )
            .dropDuplicates(["poi_id"])
            .persist(StorageLevel.DISK_ONLY)
        )
        poi_source_count = raw_pois.count()
        poi_count = poi_static.count()
        if poi_source_count != poi_count:
            raise ValueError(
                f"POI ID 不唯一：source={poi_source_count}, distinct={poi_count}"
            )
        if args.expected_poi_count is not None and poi_count != args.expected_poi_count:
            raise ValueError(f"POI 行数不一致：{poi_count} != {args.expected_poi_count}")
        print(f"[GNPR] pois={poi_count:,}, invalid_pois=0", flush=True)

        valid_users = orders.filter(
            F.col("passenger_id").isNotNull()
            & (F.length("passenger_id") > 0)
            & (~F.col("passenger_id").isin("0", "-1"))
        )
        user_history = (
            valid_users.groupBy("passenger_id")
            .agg(F.first("history_sequence", ignorenulls=True).alias("history_sequence"))
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        target_user_count = user_history.count()
        history_nonempty_user_count = user_history.filter(
            F.size("history_sequence") > 0
        ).count()
        history_events = (
            user_history.select(
                "passenger_id",
                F.explode(F.array_distinct("history_sequence")).alias("history"),
            )
            .select(
                "passenger_id",
                F.trim(F.col("history.poi_id")).alias("poi_id"),
                _timestamp(F.col("history.event_time"), F).alias("event_time"),
            )
            .filter(
                F.col("poi_id").isNotNull()
                & (F.length("poi_id") > 0)
                & F.col("event_time").isNotNull()
            )
            .select(
                "passenger_id",
                "poi_id",
                F.hour("event_time").cast("int").alias("event_hour"),
                F.lit("history").alias("source"),
            )
        )
        target_time = F.coalesce(
            _timestamp(F.col("birth_time"), F),
            _timestamp(F.col("create_time"), F),
        )
        train_targets = (
            valid_users.filter(F.col("source_dt").between(args.train_start, args.train_end))
            .select(
                "passenger_id",
                "poi_id",
                target_time.alias("event_time"),
            )
            .filter(
                F.col("poi_id").isNotNull()
                & (F.length("poi_id") > 0)
                & F.col("event_time").isNotNull()
            )
            .select(
                "passenger_id",
                "poi_id",
                F.hour("event_time").cast("int").alias("event_hour"),
                F.lit("target_train").alias("source"),
            )
        )
        interactions = history_events.unionByName(train_targets).persist(
            StorageLevel.DISK_ONLY
        )
        source_counts = {
            row["source"]: int(row["count"])
            for row in interactions.groupBy("source").count().collect()
        }
        target_train_event_count = source_counts.get("target_train", 0)
        if (
            args.expected_train_target_count is not None
            and target_train_event_count != args.expected_train_target_count
        ):
            raise ValueError(
                "训练目标订单行数不一致："
                f"{target_train_event_count} != {args.expected_train_target_count}"
            )
        interaction_count = sum(source_counts.values())
        poi_keys = poi_static.select("poi_id")
        matched = interactions.join(poi_keys, on="poi_id", how="inner").persist(
            StorageLevel.DISK_ONLY
        )
        matched_interaction_count = matched.count()
        unmatched_interaction_count = interaction_count - matched_interaction_count
        orders.unpersist(blocking=False)
        user_history.unpersist(blocking=False)
        interactions.unpersist(blocking=False)
        print(
            "[GNPR] target_users={:,}, history_nonempty_users={:,}, "
            "history_events={:,}, "
            "train_targets={:,}, matched={:,}, unmatched={:,}".format(
                target_user_count,
                history_nonempty_user_count,
                source_counts.get("history", 0),
                target_train_event_count,
                matched_interaction_count,
                unmatched_interaction_count,
            ),
            flush=True,
        )

        category_vocab = _build_indexed_vocab(
            poi_static,
            "category_code",
            "category_index",
            min(args.output_partitions, 8),
        ).persist(StorageLevel.MEMORY_AND_DISK)
        region_vocab = _build_indexed_vocab(
            poi_static,
            "plus_code_region",
            "region_index",
            min(args.output_partitions, 8),
        ).persist(StorageLevel.MEMORY_AND_DISK)
        user_vocab = _build_indexed_vocab(
            matched,
            "passenger_id",
            "user_index",
            args.output_partitions,
        ).persist(StorageLevel.MEMORY_AND_DISK)
        category_count = category_vocab.count()
        region_count = region_vocab.count()
        user_count = user_vocab.count()

        hour_counts = matched.groupBy("poi_id", "event_hour").count()
        hour_window = Window.partitionBy("poi_id").orderBy(
            F.desc("count"), F.asc("event_hour")
        )
        top_hour_rows = (
            hour_counts.withColumn("top_rank", F.row_number().over(hour_window))
            .filter(F.col("top_rank") <= args.top_k)
        )
        top_hours = top_hour_rows.groupBy("poi_id").agg(
            F.sort_array(
                F.collect_list(F.struct("top_rank", "event_hour"))
            ).alias("top_hour_items")
        ).select(
            "poi_id",
            F.expr("transform(top_hour_items, x -> x.event_hour)").alias(
                "top_visit_hours"
            ),
        )

        user_counts = matched.groupBy("poi_id", "passenger_id").count()
        user_window = Window.partitionBy("poi_id").orderBy(
            F.desc("count"), F.asc("passenger_id")
        )
        top_user_rows = (
            user_counts.withColumn("top_rank", F.row_number().over(user_window))
            .filter(F.col("top_rank") <= args.top_k)
            .join(user_vocab, on="passenger_id", how="inner")
        )
        top_users = top_user_rows.groupBy("poi_id").agg(
            F.sort_array(
                F.collect_list(
                    F.struct("top_rank", "passenger_id", "user_index")
                )
            ).alias("top_user_items")
        ).select(
            "poi_id",
            F.expr("transform(top_user_items, x -> x.passenger_id)").alias(
                "top_visitor_ids"
            ),
            F.expr("transform(top_user_items, x -> x.user_index)").alias(
                "top_visitor_indices"
            ),
        )
        interaction_totals = matched.groupBy("poi_id").agg(
            F.count("*").alias("interaction_count")
        )

        empty_ints = F.from_json(F.lit("[]"), T.ArrayType(T.IntegerType()))
        empty_longs = F.from_json(F.lit("[]"), T.ArrayType(T.LongType()))
        empty_strings = F.from_json(F.lit("[]"), T.ArrayType(T.StringType()))
        features = (
            poi_static.join(category_vocab, on="category_code", how="inner")
            .join(region_vocab, on="plus_code_region", how="inner")
            .join(top_hours, on="poi_id", how="left")
            .join(top_users, on="poi_id", how="left")
            .join(interaction_totals, on="poi_id", how="left")
            .select(
                "poi_id",
                "category_code",
                F.col("category_index").cast("long").alias("category_index"),
                "plus_code_region",
                F.col("region_index").cast("long").alias("region_index"),
                F.coalesce("top_visit_hours", empty_ints).alias("top_visit_hours"),
                F.coalesce("top_visitor_ids", empty_strings).alias("top_visitor_ids"),
                F.coalesce("top_visitor_indices", empty_longs).alias(
                    "top_visitor_indices"
                ),
                F.coalesce("interaction_count", F.lit(0)).cast("long").alias(
                    "interaction_count"
                ),
                "lng",
                "lat",
            )
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        feature_summary = features.agg(
            F.count("*").alias("feature_count"),
            F.sum(F.when(F.col("interaction_count") > 0, 1).otherwise(0)).alias(
                "behavior_poi_count"
            ),
            F.max(F.size("top_visit_hours")).alias("max_top_hours"),
            F.max(F.size("top_visitor_indices")).alias("max_top_users"),
        ).first()
        feature_count = int(feature_summary["feature_count"])
        matched.unpersist(blocking=False)
        if feature_count != poi_count:
            raise ValueError(f"POI 特征行数不一致：{feature_count} != {poi_count}")
        if int(feature_summary["max_top_hours"] or 0) > args.top_k:
            raise ValueError("top_visit_hours 超过 top-k")
        if int(feature_summary["max_top_users"] or 0) > args.top_k:
            raise ValueError("top_visitor_indices 超过 top-k")

        features_path = staging_dir / "poi_features.parquet"
        user_vocab_path = staging_dir / "user_vocab.parquet"
        category_vocab_path = staging_dir / "category_vocab.parquet"
        region_vocab_path = staging_dir / "region_vocab.parquet"
        (
            features.repartitionByRange(args.output_partitions, "poi_id")
            .sortWithinPartitions("poi_id")
            .write.mode("error").parquet(features_path.resolve().as_uri())
        )
        (
            user_vocab.repartitionByRange(args.output_partitions, "passenger_id")
            .sortWithinPartitions("passenger_id")
            .write.mode("error").parquet(user_vocab_path.resolve().as_uri())
        )
        category_vocab.orderBy("category_index").coalesce(1).write.mode(
            "error"
        ).parquet(category_vocab_path.resolve().as_uri())
        region_vocab.orderBy("region_index").coalesce(1).write.mode("error").parquet(
            region_vocab_path.resolve().as_uri()
        )

        readback = spark.read.parquet(features_path.resolve().as_uri())
        readback_count = readback.count()
        duplicate_exists = (
            readback.groupBy("poi_id").count().filter(F.col("count") != 1).limit(1).count()
        )
        if readback_count != poi_count or duplicate_exists:
            raise ValueError("POI 特征 Parquet 回读行数或唯一性检查失败")

        stats = {
            "poi_source_count": poi_source_count,
            "poi_count": poi_count,
            "order_count": order_count,
            "feature_count": feature_count,
            "readback_count": readback_count,
            "category_count": category_count,
            "region_count": region_count,
            "user_count": user_count,
            "target_user_count": target_user_count,
            "history_nonempty_user_count": history_nonempty_user_count,
            "history_event_count": source_counts.get("history", 0),
            "target_train_event_count": target_train_event_count,
            "interaction_count": interaction_count,
            "matched_interaction_count": matched_interaction_count,
            "unmatched_interaction_count": unmatched_interaction_count,
            "behavior_poi_count": int(feature_summary["behavior_poi_count"] or 0),
            "max_history_length": max_history_length,
            "max_top_hours": int(feature_summary["max_top_hours"] or 0),
            "max_top_users": int(feature_summary["max_top_users"] or 0),
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "method": "GNPR-SID",
            "data_contract": {
                "history_policy": "fixed_recent_10_once_per_passenger",
                "train_target_range": [args.train_start, args.train_end],
                "validation_and_test_targets_excluded": True,
                "plus_code_length": args.plus_code_length,
                "top_visit_hours": args.top_k,
                "top_visitors": args.top_k,
            },
            "inputs": {
                "poi_dir": str(poi_dir),
                "orders_dir": str(orders_dir),
            },
            "outputs": {
                "poi_features": "poi_features.parquet",
                "user_vocab": "user_vocab.parquet",
                "category_vocab": "category_vocab.parquet",
                "region_vocab": "region_vocab.parquet",
            },
            "stats": stats,
        }
        _write_json(staging_dir / "stats.json", stats)
        _write_json(staging_dir / "manifest.json", manifest)
        (staging_dir / "_SUCCESS").touch()
        spark.stop()
        staging_dir.rename(output_dir)
        print(f"[GNPR] completed: {output_dir}", flush=True)
        return manifest
    except Exception:
        try:
            spark.stop()
        except Exception:
            pass
        raise


def main() -> int:
    args = parse_args()
    try:
        manifest = _run(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"GNPR POI 特征构建失败：{error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
