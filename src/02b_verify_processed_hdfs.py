from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("Nhom_02_Walmart_Big_Data")
    .config("spark.ui.enabled", "true")
    .config("spark.ui.port", "4050")
    .getOrCreate()
)

path = "hdfs://localhost:9000/bigdata/walmart/processed/walmart_sales_enriched"

df = spark.read.parquet(path)

print("=" * 100)
print("VERIFY PROCESSED DATA FROM HDFS")
print("=" * 100)
print(f"Path: {path}")
print(f"So dong: {df.count()}")
print(f"So cot: {len(df.columns)}")
print("Columns:")
print(df.columns)

print("\nSchema:")
df.printSchema()

print("\nSample data:")
df.show(10, truncate=False)

if df.count() > 100000 and len(df.columns) > 10:
    print("DAT YEU CAU: Processed dataset co >100,000 records va >10 features.")
else:
    print("CHUA DAT YEU CAU.")

input("Nhan Enter de dung Spark...")
spark.stop()
