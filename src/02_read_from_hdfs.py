from pyspark.sql import SparkSession


APP_NAME = "Nhom_02_Walmart_Big_Data"

HDFS_BASE_PATH = "hdfs://localhost:9000/bigdata/walmart/raw"
HDFS_PROCESSED_PATH = "hdfs://localhost:9000/bigdata/walmart/processed/walmart_sales_enriched"

TRAIN_PATH = f"{HDFS_BASE_PATH}/train.csv"
FEATURES_PATH = f"{HDFS_BASE_PATH}/features.csv"
STORES_PATH = f"{HDFS_BASE_PATH}/stores.csv"
TEST_PATH = f"{HDFS_BASE_PATH}/test.csv"


def print_basic_info(name, df):
    print("=" * 100)
    print(f"DATASET FROM HDFS: {name}")
    print("=" * 100)

    rows = df.count()
    cols = len(df.columns)

    print(f"So dong: {rows}")
    print(f"So cot: {cols}")
    print(f"Danh sach cot: {df.columns}")

    print("\nSchema:")
    df.printSchema()

    print("\n5 dong dau tien:")
    df.show(5, truncate=False)


def print_missing_values(spark, view_name, df):
    print("=" * 100)
    print(f"MISSING VALUES: {view_name}")
    print("=" * 100)

    df.createOrReplaceTempView(view_name)

    expressions = []
    for col_name in df.columns:
        expressions.append(
            f"SUM(CASE WHEN `{col_name}` IS NULL THEN 1 ELSE 0 END) AS `{col_name}`"
        )

    query = f"""
    SELECT
        {", ".join(expressions)}
    FROM {view_name}
    """

    spark.sql(query).show(truncate=False)


if __name__ == "__main__":
    spark = (
        SparkSession.builder
        .appName(APP_NAME)
        .config("spark.ui.enabled", "true")
        .config("spark.ui.port", "4050")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    print("=" * 100)
    print("SPARK STARTED")
    print("=" * 100)
    print(f"AppName: {APP_NAME}")
    print("Spark UI: http://localhost:4050")
    print(f"HDFS raw path: {HDFS_BASE_PATH}")

    print("=" * 100)
    print("READING CSV FILES FROM HDFS")
    print("=" * 100)

    train = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("nullValue", "NA")
        .csv(TRAIN_PATH)
    )

    features = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("nullValue", "NA")
        .csv(FEATURES_PATH)
    )

    stores = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("nullValue", "NA")
        .csv(STORES_PATH)
    )

    test = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("nullValue", "NA")
        .csv(TEST_PATH)
    )

    print_basic_info("train.csv", train)
    print_basic_info("features.csv", features)
    print_basic_info("stores.csv", stores)
    print_basic_info("test.csv", test)

    print_missing_values(spark, "train", train)
    print_missing_values(spark, "features", features)
    print_missing_values(spark, "stores", stores)
    print_missing_values(spark, "test", test)

    train.createOrReplaceTempView("train")
    features.createOrReplaceTempView("features")
    stores.createOrReplaceTempView("stores")
    test.createOrReplaceTempView("test")

    print("=" * 100)
    print("CREATING walmart_sales_enriched BY SPARK SQL JOIN")
    print("=" * 100)

    walmart_sales_enriched = spark.sql("""
        SELECT
            t.Store,
            t.Dept,
            CAST(t.Date AS DATE) AS Date,
            CAST(t.Weekly_Sales AS DOUBLE) AS Weekly_Sales,
            t.IsHoliday,

            TRY_CAST(f.Temperature AS DOUBLE) AS Temperature,
            TRY_CAST(f.Fuel_Price AS DOUBLE) AS Fuel_Price,

            COALESCE(TRY_CAST(f.MarkDown1 AS DOUBLE), 0.0) AS MarkDown1,
            COALESCE(TRY_CAST(f.MarkDown2 AS DOUBLE), 0.0) AS MarkDown2,
            COALESCE(TRY_CAST(f.MarkDown3 AS DOUBLE), 0.0) AS MarkDown3,
            COALESCE(TRY_CAST(f.MarkDown4 AS DOUBLE), 0.0) AS MarkDown4,
            COALESCE(TRY_CAST(f.MarkDown5 AS DOUBLE), 0.0) AS MarkDown5,

            TRY_CAST(f.CPI AS DOUBLE) AS CPI,
            TRY_CAST(f.Unemployment AS DOUBLE) AS Unemployment,

            s.Type,
            CAST(s.Size AS INT) AS Size,

            YEAR(CAST(t.Date AS DATE)) AS year,
            MONTH(CAST(t.Date AS DATE)) AS month,
            WEEKOFYEAR(CAST(t.Date AS DATE)) AS week_of_year,

            COALESCE(TRY_CAST(f.MarkDown1 AS DOUBLE), 0.0)
            + COALESCE(TRY_CAST(f.MarkDown2 AS DOUBLE), 0.0)
            + COALESCE(TRY_CAST(f.MarkDown3 AS DOUBLE), 0.0)
            + COALESCE(TRY_CAST(f.MarkDown4 AS DOUBLE), 0.0)
            + COALESCE(TRY_CAST(f.MarkDown5 AS DOUBLE), 0.0) AS markdown_total

        FROM train t
        LEFT JOIN features f
            ON t.Store = f.Store
            AND t.Date = f.Date
        LEFT JOIN stores s
            ON t.Store = s.Store
    """)

    walmart_sales_enriched.createOrReplaceTempView("walmart_sales_enriched")

    enriched_rows = walmart_sales_enriched.count()
    enriched_cols = len(walmart_sales_enriched.columns)

    print("=" * 100)
    print("RESULT AFTER JOIN")
    print("=" * 100)
    print(f"So dong sau JOIN: {enriched_rows}")
    print(f"So cot sau JOIN: {enriched_cols}")

    print("\nSchema cua walmart_sales_enriched:")
    walmart_sales_enriched.printSchema()

    print("\n10 dong dau tien cua walmart_sales_enriched:")
    walmart_sales_enriched.show(10, truncate=False)

    print("=" * 100)
    print("CHECK REQUIREMENT: >100,000 RECORDS AND >10 FEATURES")
    print("=" * 100)

    if enriched_rows > 100000 and enriched_cols > 10:
        print("DAT YEU CAU.")
        print(f"Dataset sau JOIN co {enriched_rows} records va {enriched_cols} features.")
    else:
        print("CHUA DAT YEU CAU.")
        print(f"Dataset sau JOIN co {enriched_rows} records va {enriched_cols} features.")

    print("=" * 100)
    print("CHECK MISSING VALUES AFTER JOIN")
    print("=" * 100)

    missing_query_parts = []
    for col_name in walmart_sales_enriched.columns:
        missing_query_parts.append(
            f"SUM(CASE WHEN `{col_name}` IS NULL THEN 1 ELSE 0 END) AS `{col_name}`"
        )

    missing_query = f"""
    SELECT
        {", ".join(missing_query_parts)}
    FROM walmart_sales_enriched
    """

    spark.sql(missing_query).show(truncate=False)

    print("=" * 100)
    print("SUMMARY TABLE FOR REPORT")
    print("=" * 100)

    summary = spark.sql(f"""
        SELECT 'train.csv' AS file_name, COUNT(*) AS rows, {len(train.columns)} AS columns FROM train
        UNION ALL
        SELECT 'features.csv' AS file_name, COUNT(*) AS rows, {len(features.columns)} AS columns FROM features
        UNION ALL
        SELECT 'stores.csv' AS file_name, COUNT(*) AS rows, {len(stores.columns)} AS columns FROM stores
        UNION ALL
        SELECT 'test.csv' AS file_name, COUNT(*) AS rows, {len(test.columns)} AS columns FROM test
        UNION ALL
        SELECT 'walmart_sales_enriched' AS file_name, COUNT(*) AS rows, {len(walmart_sales_enriched.columns)} AS columns FROM walmart_sales_enriched
    """)

    summary.show(truncate=False)

    print("=" * 100)
    print("WRITE walmart_sales_enriched TO HDFS AS PARQUET")
    print("=" * 100)

    (
        walmart_sales_enriched
        .write
        .mode("overwrite")
        .parquet(HDFS_PROCESSED_PATH)
    )

    print(f"Saved successfully to: {HDFS_PROCESSED_PATH}")

    print("=" * 100)
    print("DONE")
    print("=" * 100)

    input("Nhan Enter de dung Spark va dong Spark UI...")

    spark.stop()
