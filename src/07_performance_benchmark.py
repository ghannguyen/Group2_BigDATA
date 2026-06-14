import time

from pyspark.storagelevel import StorageLevel
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


APP_NAME = "Nhom_02_Walmart_Big_Data_SQL_Cache_Benchmark"
PROCESSED_PATH = (
    "hdfs://localhost:9000/bigdata/walmart/processed/"
    "walmart_sales_enriched"
)
BENCHMARK_OUTPUT_PATH = (
    "hdfs://localhost:9000/bigdata/walmart/benchmarks/"
    "sql_cache_benchmark"
)


def build_query_b(df, view_name):
    df.createOrReplaceTempView(view_name)
    return df.sparkSession.sql(
        f"""
        WITH monthly_sales AS (
            SELECT
                Year,
                Month,
                SUM(Weekly_Sales) AS monthly_revenue
            FROM {view_name}
            GROUP BY Year, Month
        ),
        yearly_sales AS (
            SELECT
                Year,
                SUM(monthly_revenue) AS yearly_revenue
            FROM monthly_sales
            GROUP BY Year
        )
        SELECT
            m.Year,
            m.Month,
            ROUND(m.monthly_revenue, 2) AS monthly_revenue,
            ROUND(y.yearly_revenue, 2) AS yearly_revenue,
            ROUND(
                (m.monthly_revenue / y.yearly_revenue) * 100,
                4
            ) AS monthly_revenue_percent
        FROM monthly_sales m
        INNER JOIN yearly_sales y
            ON m.Year = y.Year
        ORDER BY m.Year, m.Month
        """
    )


def run_three_queries(df, label: str) -> dict:
    print("\n" + "=" * 100)
    print(f"RUN THREE QUERIES: {label}")
    print("=" * 100)

    print("\n[QUERY A] Rank stores by average Weekly_Sales per year")
    start_a = time.time()

    store_year_sales = (
        df.groupBy("Year", "Store")
        .agg(F.avg("Weekly_Sales").alias("avg_sales"))
    )
    ranking_window = Window.partitionBy("Year").orderBy(F.desc("avg_sales"))
    query_a = (
        store_year_sales
        .withColumn("store_rank", F.rank().over(ranking_window))
        .orderBy("Year", "store_rank", "Store")
    )
    query_a.show(5, truncate=False)

    time_a = time.time() - start_a
    print(f"Query A time ({label}): {time_a:.4f} seconds")

    print("\n[QUERY B] Monthly revenue percentage contribution by year")
    start_b = time.time()

    view_name = f"walmart_sales_{label.lower()}"
    query_b = build_query_b(df, view_name)
    query_b.show(5, truncate=False)

    time_b = time.time() - start_b
    print(f"Query B time ({label}): {time_b:.4f} seconds")

    print("\n[QUERY C] Revenue summary using CUBE(Store, IsHoliday)")
    start_c = time.time()

    query_c = (
        df.cube("Store", "IsHoliday")
        .agg(
            F.sum("Weekly_Sales").alias("total_weekly_sales"),
            F.avg("Weekly_Sales").alias("avg_weekly_sales"),
            F.count(F.lit(1)).alias("record_count"),
        )
        .orderBy(
            F.col("Store").asc_nulls_last(),
            F.col("IsHoliday").asc_nulls_last(),
        )
    )
    query_c.show(5, truncate=False)

    time_c = time.time() - start_c
    total_time = time_a + time_b + time_c

    print(f"Query C time ({label}): {time_c:.4f} seconds")
    print(f"Total time ({label}): {total_time:.4f} seconds")

    return {
        "label": label,
        "A": time_a,
        "B": time_b,
        "C": time_c,
        "total": total_time,
    }


def print_comparison(result_no_cache, result_cached):
    print("\n" + "=" * 100)
    print("PERFORMANCE COMPARISON")
    print("=" * 100)
    print(
        f"| {'Query':<8} | {'No Cache (s)':>14} | "
        f"{'Cached (s)':>12} | {'Speedup %':>10} |"
    )
    print(f"|{'-' * 10}|{'-' * 16}|{'-' * 14}|{'-' * 12}|")

    for query_name in ("A", "B", "C", "total"):
        no_cache_time = result_no_cache[query_name]
        cached_time = result_cached[query_name]
        improvement = (
            ((no_cache_time - cached_time) / no_cache_time) * 100
            if no_cache_time > 0
            else 0.0
        )
        display_name = query_name.upper() if query_name != "total" else "TOTAL"
        print(
            f"| {display_name:<8} | {no_cache_time:>14.4f} | "
            f"{cached_time:>12.4f} | {improvement:>9.2f}% |"
        )


if __name__ == "__main__":
    print("=" * 100)
    print("WALMART SPARK SQL PERFORMANCE BENCHMARK")
    print("=" * 100)

    print("\n[STEP 1] Create SparkSession")
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName(APP_NAME)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    print("Spark AppName:", APP_NAME)
    print("Spark master:", spark.sparkContext.master)
    print(
        "spark.sql.shuffle.partitions:",
        spark.conf.get("spark.sql.shuffle.partitions"),
    )

    df_cached = None

    try:
        print("\n[STEP 2] Read processed Parquet from HDFS without caching")
        print("Input path:", PROCESSED_PATH)
        df_raw = spark.read.parquet(PROCESSED_PATH)
        print("Columns:", len(df_raw.columns))
        print("Partitions:", df_raw.rdd.getNumPartitions())

        print("\n[STEP 3] Explain Query B before cache")
        print("=" * 100)
        query_b_before_cache = build_query_b(
            df_raw,
            "walmart_sales_explain_no_cache",
        )
        query_b_before_cache.explain(True)

        print("\n[STEP 4] Run benchmark without cache")
        result_no_cache = run_three_queries(df_raw, "NO_CACHE")

        print("\n[STEP 5] Persist dataset with MEMORY_AND_DISK")
        df_cached = df_raw.persist(StorageLevel.MEMORY_AND_DISK)

        warmup_start = time.time()
        cached_row_count = df_cached.count()
        warmup_time = time.time() - warmup_start

        print("Cache warm-up completed.")
        print("Cached rows:", cached_row_count)
        print(f"Cache warm-up time: {warmup_time:.4f} seconds")

        print("\n[STEP 6] Explain Query B after cache")
        print("=" * 100)
        query_b_after_cache = build_query_b(
            df_cached,
            "walmart_sales_explain_with_cache",
        )
        query_b_after_cache.explain(True)

        print("\n[STEP 7] Run benchmark with cache")
        result_cached = run_three_queries(df_cached, "WITH_CACHE")

        print_comparison(result_no_cache, result_cached)

        print("\n[STEP 8] Save benchmark results to HDFS as CSV")
        benchmark_results = [
            {
                "label": result_no_cache["label"],
                "query_A": result_no_cache["A"],
                "query_B": result_no_cache["B"],
                "query_C": result_no_cache["C"],
                "total": result_no_cache["total"],
            },
            {
                "label": result_cached["label"],
                "query_A": result_cached["A"],
                "query_B": result_cached["B"],
                "query_C": result_cached["C"],
                "total": result_cached["total"],
            },
        ]
        benchmark_df = spark.createDataFrame(benchmark_results).select(
            "label",
            "query_A",
            "query_B",
            "query_C",
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

        print("\n[STEP 9] Unpersist cached dataset")
        df_cached.unpersist()
        df_cached = None
        print("Dataset unpersisted successfully.")

        print("\n" + "=" * 100)
        print("PERFORMANCE BENCHMARK COMPLETED")
        print("=" * 100)
    finally:
        if df_cached is not None:
            df_cached.unpersist()
        spark.stop()
