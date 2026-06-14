import time

from pyspark.storagelevel import StorageLevel
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


APP_NAME = "Walmart_Partition_Benchmark"
PROCESSED_PATH = (
    "hdfs://localhost:9000/bigdata/walmart/processed/"
    "walmart_sales_enriched"
)
BENCHMARK_OUTPUT_PATH = (
    "hdfs://localhost:9000/bigdata/walmart/benchmarks/"
    "partition_benchmark"
)


def build_aggregation_1(df):
    return (
        df.groupBy("Store")
        .agg(
            F.sum("Weekly_Sales").alias("total_weekly_sales"),
            F.avg("Weekly_Sales").alias("avg_weekly_sales"),
            F.max("Weekly_Sales").alias("max_weekly_sales"),
        )
        .orderBy("Store")
    )


def build_aggregation_2(df):
    monthly_sales = (
        df.groupBy("Store", "Year", "Month")
        .agg(F.sum("Weekly_Sales").alias("total_weekly_sales"))
    )
    rank_window = Window.partitionBy("Store").orderBy(
        F.desc("total_weekly_sales"),
        F.asc("Year"),
        F.asc("Month"),
    )
    return (
        monthly_sales
        .withColumn("sales_rank", F.rank().over(rank_window))
        .orderBy("Store", "sales_rank", "Year", "Month")
    )


def build_aggregation_3(df):
    return (
        df.groupBy("Type", "IsHoliday")
        .agg(
            F.avg("Weekly_Sales").alias("avg_weekly_sales"),
            F.stddev("Weekly_Sales").alias("stddev_weekly_sales"),
        )
        .orderBy("Type", "IsHoliday")
    )


def run_aggregation_benchmark(df, label: str) -> dict:
    print("\n" + "=" * 100)
    print(f"AGGREGATION BENCHMARK: {label}")
    print("=" * 100)

    total_start = time.perf_counter()

    print("\n[AGG 1] Sales statistics by Store")
    agg1_start = time.perf_counter()
    build_aggregation_1(df).show(5, truncate=False)
    agg1_time = time.perf_counter() - agg1_start
    print(f"Agg 1 time ({label}): {agg1_time:.4f} seconds")

    print("\n[AGG 2] Monthly sales rank within each Store")
    agg2_start = time.perf_counter()
    build_aggregation_2(df).show(5, truncate=False)
    agg2_time = time.perf_counter() - agg2_start
    print(f"Agg 2 time ({label}): {agg2_time:.4f} seconds")

    print("\n[AGG 3] Sales statistics by Type and IsHoliday")
    agg3_start = time.perf_counter()
    build_aggregation_3(df).show(5, truncate=False)
    agg3_time = time.perf_counter() - agg3_start
    print(f"Agg 3 time ({label}): {agg3_time:.4f} seconds")

    total_time = time.perf_counter() - total_start
    print(f"Total time ({label}): {total_time:.4f} seconds")

    return {
        "label": label,
        "agg1": agg1_time,
        "agg2": agg2_time,
        "agg3": agg3_time,
        "total": total_time,
        "num_partitions": df.rdd.getNumPartitions(),
    }


def print_comparison_table(result_default, result_repart):
    print("\n" + "=" * 100)
    print("PARTITION STRATEGY COMPARISON")
    print("=" * 100)
    print(
        f"| {'Strategy':<20} | {'Partitions':>10} | {'Agg1(s)':>10} | "
        f"{'Agg2(s)':>10} | {'Agg3(s)':>10} | {'Total(s)':>10} |"
    )
    print(
        f"|{'-' * 22}|{'-' * 12}|{'-' * 12}|"
        f"{'-' * 12}|{'-' * 12}|{'-' * 12}|"
    )

    display_rows = (
        ("DEFAULT", result_default),
        ("REPARTITION_8", result_repart),
    )
    for display_label, result in display_rows:
        print(
            f"| {display_label:<20} | {result['num_partitions']:>10} | "
            f"{result['agg1']:>10.4f} | {result['agg2']:>10.4f} | "
            f"{result['agg3']:>10.4f} | {result['total']:>10.4f} |"
        )


def print_agg2_physical_plans(df_default, df_repart):
    print("\n" + "=" * 100)
    print("AGG 2 PHYSICAL PLAN COMPARISON")
    print("=" * 100)

    print("\n" + "#" * 100)
    print("# LEFT / DEFAULT PARTITION STRATEGY")
    print("#" * 100)
    build_aggregation_2(df_default).explain(True)

    print("\n" + "#" * 100)
    print("# RIGHT / REPARTITION(8, STORE) STRATEGY")
    print("#" * 100)
    build_aggregation_2(df_repart).explain(True)


def print_interpretation(result_default, result_repart):
    default_total = result_default["total"]
    repart_total = result_repart["total"]
    difference = default_total - repart_total
    percent_change = (
        abs(difference) / default_total * 100.0
        if default_total > 0
        else 0.0
    )

    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    print(f"DEFAULT total aggregation time: {default_total:.4f} seconds.")
    print(
        "REPARTITION_8_STORE total aggregation time: "
        f"{repart_total:.4f} seconds."
    )

    if repart_total < default_total:
        print(
            "Repartition helped: the measured aggregation phase was "
            f"{percent_change:.2f}% faster."
        )
        print(
            "Partitioning by Store can reduce repeated Store-key data movement "
            "for Store-based aggregations and the per-Store window."
        )
    elif repart_total > default_total:
        print(
            "Repartition hurt: the measured aggregation phase was "
            f"{percent_change:.2f}% slower."
        )
        print(
            "The extra partitioning and task overhead outweighed any reduction "
            "in Store-key data movement for this dataset and local Spark setup."
        )
    else:
        print("Both strategies had the same measured total aggregation time.")
        print(
            "At this scale, repartitioning by Store produced no measurable "
            "benefit for the tested workload."
        )

    print(
        "The one-time cache warm-up and repartition materialization times are "
        "excluded from the aggregation totals."
    )


if __name__ == "__main__":
    print("=" * 100)
    print("WALMART PARTITION STRATEGY BENCHMARK")
    print("=" * 100)

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName(APP_NAME)
        .config("spark.ui.port", "4050")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    df_default = None
    df_repart = None

    try:
        print("\n[STEP 1] Read processed Parquet from HDFS")
        print("Input path:", PROCESSED_PATH)
        df = spark.read.parquet(PROCESSED_PATH)

        required_columns = {
            "Store",
            "Weekly_Sales",
            "Year",
            "Month",
            "Type",
            "IsHoliday",
        }
        missing_columns = sorted(required_columns.difference(df.columns))
        if missing_columns:
            raise ValueError(
                "Processed dataset is missing required columns: "
                + ", ".join(missing_columns)
            )

        print("Columns:", len(df.columns))
        print(f"Default partitions: {df.rdd.getNumPartitions()}")

        print("\n[STEP 2] Materialize DEFAULT strategy")
        df_default = df.persist(StorageLevel.MEMORY_AND_DISK)
        default_warmup_start = time.perf_counter()
        default_rows = df_default.count()
        default_warmup_time = time.perf_counter() - default_warmup_start
        print("DEFAULT rows:", default_rows)
        print(f"DEFAULT warm-up time: {default_warmup_time:.4f} seconds")

        result_default = run_aggregation_benchmark(df_default, "DEFAULT")

        print("\n[STEP 3] Repartition by Store and materialize")
        df_repart = df.repartition(8, "Store").persist(
            StorageLevel.MEMORY_AND_DISK
        )
        print(f"After repartition: {df_repart.rdd.getNumPartitions()}")
        repart_warmup_start = time.perf_counter()
        repart_rows = df_repart.count()
        repart_warmup_time = time.perf_counter() - repart_warmup_start
        print("REPARTITION_8_STORE rows:", repart_rows)
        print(
            "REPARTITION_8_STORE warm-up time: "
            f"{repart_warmup_time:.4f} seconds"
        )

        result_repart = run_aggregation_benchmark(
            df_repart,
            "REPARTITION_8_STORE",
        )

        print_comparison_table(result_default, result_repart)
        print_agg2_physical_plans(df_default, df_repart)

        print("\n[STEP 4] Save benchmark results to HDFS as CSV")
        benchmark_df = spark.createDataFrame(
            [result_default, result_repart]
        ).select(
            "label",
            "num_partitions",
            "agg1",
            "agg2",
            "agg3",
            "total",
        )
        benchmark_df.show(truncate=False)
        (
            benchmark_df.coalesce(1)
            .write
            .mode("overwrite")
            .option("header", True)
            .csv(BENCHMARK_OUTPUT_PATH)
        )
        print("Benchmark output path:", BENCHMARK_OUTPUT_PATH)

        print_interpretation(result_default, result_repart)

        print("\n" + "=" * 100)
        print("PARTITION BENCHMARK COMPLETED")
        print("=" * 100)
    finally:
        if df_repart is not None:
            df_repart.unpersist()
        if df_default is not None:
            df_default.unpersist()
        spark.stop()
