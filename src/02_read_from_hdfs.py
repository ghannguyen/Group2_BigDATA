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
    print("TEMP VIEWS CREATED SUCCESSFULLY")
    print("=" * 100)

    print("Available views:")
    print("- train")
    print("- features")
    print("- stores")
    print("- test")

    print("=" * 100)
    print("DATA READING AND EXPLORATION COMPLETED")
    print("=" * 100)

    print("Summary:")
    print(f"train.csv    : {train.count()} rows | {len(train.columns)} columns")
    print(f"features.csv : {features.count()} rows | {len(features.columns)} columns")
    print(f"stores.csv   : {stores.count()} rows | {len(stores.columns)} columns")
    print(f"test.csv     : {test.count()} rows | {len(test.columns)} columns")

    print("=" * 100)
    print("DONE")
    print("=" * 100)

    input("Nhan Enter de dung Spark va dong Spark UI...")

    spark.stop()