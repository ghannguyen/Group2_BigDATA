from pyspark.sql import SparkSession

# ============================================================
# VERIFY PROCESSED DATASET FROM HDFS
# Input : walmart_sales_enriched (Parquet)
# Purpose: Validate output after preprocessing pipeline
# ============================================================

spark = (
    SparkSession.builder
    .appName("Nhom_02_Walmart_Verify_Processed_Data")
    .config("spark.ui.enabled", "true")
    .config("spark.ui.port", "4050")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# ============================================================
# READ PROCESSED PARQUET
# ============================================================

path = "hdfs://localhost:9000/bigdata/walmart/processed/walmart_sales_enriched"

df = spark.read.parquet(path)

# ============================================================
# DATASET OVERVIEW
# ============================================================

print("=" * 100)
print("VERIFY PROCESSED DATASET FROM HDFS")
print("=" * 100)

print("Dataset Path:")
print(path)

print("\nDataset Summary:")
print(f"Rows    : {df.count()}")
print(f"Columns : {len(df.columns)}")

print("\nColumn List:")
for c in df.columns:
    print("-", c)

# ============================================================
# SCHEMA
# ============================================================

print("\n" + "=" * 100)
print("DATASET SCHEMA")
print("=" * 100)

df.printSchema()

# ============================================================
# SAMPLE DATA
# ============================================================

print("\n" + "=" * 100)
print("SAMPLE DATA")
print("=" * 100)

df.show(10, truncate=False)

# ============================================================
# VERIFY REQUIREMENT
# ============================================================

print("\n" + "=" * 100)
print("REQUIREMENT CHECK")
print("=" * 100)

row_count = df.count()
column_count = len(df.columns)

if row_count > 100000:
    print(f"[PASS] Record count > 100,000 ({row_count:,})")
else:
    print(f"[FAIL] Record count <= 100,000 ({row_count:,})")

if column_count > 10:
    print(f"[PASS] Feature count > 10 ({column_count})")
else:
    print(f"[FAIL] Feature count <= 10 ({column_count})")

# ============================================================
# CHECK NULL VALUES
# ============================================================

print("\n" + "=" * 100)
print("NULL VALUE CHECK")
print("=" * 100)

from pyspark.sql.functions import col, sum as spark_sum

df.select([
    spark_sum(col(c).isNull().cast("int")).alias(c)
    for c in df.columns
]).show(truncate=False)

# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 100)
print("VERIFY COMPLETED")
print("=" * 100)

input("Nhan Enter de dung Spark...")

spark.stop()