from pyspark.sql import SparkSession


APP_NAME = "Nhom_02_Walmart_Check_Dataset"

HDFS_RAW_PATH = "hdfs://localhost:9000/bigdata/walmart/raw"
HDFS_PROCESSED_PATH = "hdfs://localhost:9000/bigdata/walmart/processed/walmart_sales_enriched"

TRAIN_PATH = f"{HDFS_RAW_PATH}/train.csv"
FEATURES_PATH = f"{HDFS_RAW_PATH}/features.csv"
STORES_PATH = f"{HDFS_RAW_PATH}/stores.csv"
TEST_PATH = f"{HDFS_RAW_PATH}/test.csv"


def print_dataset_info(name, df):
    print("=" * 100)
    print(f"DATASET: {name}")
    print("=" * 100)

    row_count = df.count()
    col_count = len(df.columns)

    print(f"So dong: {row_count}")
    print(f"So cot: {col_count}")
    print(f"Danh sach cot: {df.columns}")

    print("\nSchema:")
    df.printSchema()

    print("\n5 dong dau tien:")
    df.show(5, truncate=False)

    return row_count, col_count


def check_missing_values(spark, view_name, df):
    print("=" * 100)
    print(f"MISSING VALUES: {view_name}")
    print("=" * 100)

    df.createOrReplaceTempView(view_name)

    missing_expr = []
    for col_name in df.columns:
        missing_expr.append(
            f"SUM(CASE WHEN `{col_name}` IS NULL THEN 1 ELSE 0 END) AS `{col_name}_null`"
        )

    query = f"""
        SELECT
            {", ".join(missing_expr)}
        FROM {view_name}
    """

    spark.sql(query).show(truncate=False)


if __name__ == "__main__":
    spark = (
        SparkSession.builder
        .appName(APP_NAME)
        .config("spark.ui.port", "4050")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    print("=" * 100)
    print("CHECK WALMART DATASET FROM HDFS")
    print("=" * 100)
    print("AppName:", APP_NAME)
    print("Spark UI: http://localhost:4050")
    print("Raw HDFS path:", HDFS_RAW_PATH)
    print("Processed HDFS path:", HDFS_PROCESSED_PATH)

    print("=" * 100)
    print("READ RAW CSV FILES FROM HDFS")
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

    train_rows, train_cols = print_dataset_info("train.csv", train)
    features_rows, features_cols = print_dataset_info("features.csv", features)
    stores_rows, stores_cols = print_dataset_info("stores.csv", stores)
    test_rows, test_cols = print_dataset_info("test.csv", test)

    check_missing_values(spark, "train", train)
    check_missing_values(spark, "features", features)
    check_missing_values(spark, "stores", stores)
    check_missing_values(spark, "test", test)

    print("=" * 100)
    print("READ PROCESSED DATASET FROM HDFS")
    print("=" * 100)

    processed = spark.read.parquet(HDFS_PROCESSED_PATH)

    processed_rows, processed_cols = print_dataset_info(
        "walmart_sales_enriched", processed
    )

    check_missing_values(spark, "walmart_sales_enriched", processed)

    print("=" * 100)
    print("CHECK DATASET REQUIREMENT")
    print("=" * 100)

    print("Yeu cau de bai:")
    print("- So dong > 100,000")
    print("- So features/cot > 10")

    print("\nKet qua bang phan tich chinh:")
    print(f"- walmart_sales_enriched rows: {processed_rows}")
    print(f"- walmart_sales_enriched columns/features: {processed_cols}")

    if processed_rows > 100000 and processed_cols > 10:
        print("\nKET LUAN: DAT YEU CAU DATASET.")
        print(
            f"Bang walmart_sales_enriched co {processed_rows} records "
            f"va {processed_cols} features, dap ung yeu cau >100,000 records va >10 features."
        )
    else:
        print("\nKET LUAN: CHUA DAT YEU CAU DATASET.")
        print(
            f"Bang walmart_sales_enriched chi co {processed_rows} records "
            f"va {processed_cols} features."
        )

    print("=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)

    summary_data = [
        ("train.csv", train_rows, train_cols, "Raw fact table - weekly sales"),
        ("features.csv", features_rows, features_cols, "Raw feature table - economic and markdown data"),
        ("stores.csv", stores_rows, stores_cols, "Raw store table - store type and size"),
        ("test.csv", test_rows, test_cols, "Raw test table - no Weekly_Sales"),
        ("walmart_sales_enriched", processed_rows, processed_cols, "Processed table for Spark SQL and MLlib"),
    ]

    print(f"{'file_name':<28} {'rows':>12} {'columns':>10}  role")
    print("-" * 100)
    for file_name, rows, columns, role in summary_data:
        print(f"{file_name:<28} {rows:>12} {columns:>10}  {role}")

    print("=" * 100)
    print("DONE")
    print("=" * 100)

    input("Nhan Enter de dung Spark...")
    spark.stop()
