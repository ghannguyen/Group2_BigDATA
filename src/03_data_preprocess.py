from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    to_date,
    year,
    month,
    weekofyear,
    sum as spark_sum,
    coalesce,
    lit
)

# ============================================================
# WALMART DATA PREPROCESSING PIPELINE
# Input  : Raw CSV files on HDFS
# Output : walmart_sales_enriched (Parquet on HDFS)
# ============================================================

spark = (
    SparkSession.builder
    .appName("Nhom_02_Walmart_Big_Data_Preprocess")
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("=" * 100)
print("DATA PREPROCESSING PIPELINE")
print("=" * 100)

# ============================================================
# 1. READ RAW DATA FROM HDFS
# ============================================================

base_path = "hdfs://localhost:9000/bigdata/walmart/raw"

train = spark.read.csv(
    f"{base_path}/train.csv",
    header=True,
    inferSchema=True,
    nullValue="NA"
)

features = spark.read.csv(
    f"{base_path}/features.csv",
    header=True,
    inferSchema=True,
    nullValue="NA"
)

stores = spark.read.csv(
    f"{base_path}/stores.csv",
    header=True,
    inferSchema=True,
    nullValue="NA"
)

print("=" * 100)
print("RAW DATASET SHAPE")
print("=" * 100)

print("train rows:", train.count(), "| columns:", len(train.columns))
print("features rows:", features.count(), "| columns:", len(features.columns))
print("stores rows:", stores.count(), "| columns:", len(stores.columns))

# ============================================================
# 2. JOIN DATASETS
# ============================================================

print("=" * 100)
print("JOIN DATASETS")
print("=" * 100)

df = (
    train.join(
        features,
        ["Store", "Date", "IsHoliday"],
        "left"
    )
    .join(
        stores,
        ["Store"],
        "left"
    )
)

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

print("=" * 100)
print("FEATURE ENGINEERING")
print("=" * 100)

df = (
    df.withColumn(
        "Date",
        to_date(col("Date"), "yyyy-MM-dd")
    )
    .withColumn(
        "Year",
        year(col("Date"))
    )
    .withColumn(
        "Month",
        month(col("Date"))
    )
    .withColumn(
        "WeekOfYear",
        weekofyear(col("Date"))
    )
    .withColumn(
        "markdown_total",
        coalesce(col("MarkDown1"), lit(0.0))
        + coalesce(col("MarkDown2"), lit(0.0))
        + coalesce(col("MarkDown3"), lit(0.0))
        + coalesce(col("MarkDown4"), lit(0.0))
        + coalesce(col("MarkDown5"), lit(0.0))
    )
)

# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

print("=" * 100)
print("HANDLE MISSING VALUES")
print("=" * 100)

df = df.fillna({
    "MarkDown1": 0.0,
    "MarkDown2": 0.0,
    "MarkDown3": 0.0,
    "MarkDown4": 0.0,
    "MarkDown5": 0.0,
    "CPI": 0.0,
    "Unemployment": 0.0,
    "Temperature": 0.0,
    "Fuel_Price": 0.0
})

# ============================================================
# 5. VERIFY DATASET
# ============================================================

print("=" * 100)
print("FINAL DATASET INFORMATION")
print("=" * 100)

print("Rows:", df.count())
print("Columns:", len(df.columns))

df.printSchema()

print("=" * 100)
print("MISSING VALUES AFTER CLEANING")
print("=" * 100)

df.select([
    spark_sum(
        col(c).isNull().cast("int")
    ).alias(c)
    for c in df.columns
]).show(truncate=False)

# ============================================================
# 6. SAVE TO HDFS AS PARQUET
# ============================================================

output_path = (
    "hdfs://localhost:9000/"
    "bigdata/walmart/processed/"
    "walmart_sales_enriched"
)

print("=" * 100)
print("SAVE walmart_sales_enriched TO HDFS")
print("=" * 100)

(
    df.write
    .mode("overwrite")
    .parquet(output_path)
)

print("Output path:", output_path)
print("Rows:", df.count())
print("Columns:", len(df.columns))

print("=" * 100)
print("DATA PREPROCESSING COMPLETED")
print("=" * 100)

spark.stop()