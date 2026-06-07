from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, year, month, weekofyear, sum as spark_sum

# Create Spark session and connect to HDFS
spark = SparkSession.builder \
    .appName("Walmart_Preprocess") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .getOrCreate()

# 1. Read raw CSV files from HDFS
# nullValue="NA" helps Spark understand NA as missing value
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

# 2. Print raw dataset shape
print("===== RAW DATASET SHAPE =====")
print("train rows:", train.count(), "| columns:", len(train.columns))
print("features rows:", features.count(), "| columns:", len(features.columns))
print("stores rows:", stores.count(), "| columns:", len(stores.columns))

# 3. Join datasets
# train is the main table because it contains Weekly_Sales
# Join train with features by Store, Date, IsHoliday
# Join with stores by Store
df = train.join(features, ["Store", "Date", "IsHoliday"], "left") \
          .join(stores, ["Store"], "left")

# 4. Convert Date and create time features
df = df.withColumn("Date", to_date(col("Date"), "yyyy-MM-dd")) \
       .withColumn("Year", year(col("Date"))) \
       .withColumn("Month", month(col("Date"))) \
       .withColumn("WeekOfYear", weekofyear(col("Date")))

# 5. Handle missing values
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

# 6. Check final dataset
print("===== FINAL DATASET SHAPE =====")
print("Rows:", df.count())
print("Columns:", len(df.columns))
df.printSchema()

print("===== MISSING VALUES AFTER CLEANING =====")
df.select([
    spark_sum(col(c).isNull().cast("int")).alias(c)
    for c in df.columns
]).show(truncate=False)

# 7. Save processed dataset to HDFS as Parquet
# Parquet is recommended for Spark because it is faster and better for SQL/MLlib
output_path = "hdfs://localhost:9000/bigdata/walmart/processed/walmart_sales_enriched"

df.write \
  .mode("overwrite") \
  .parquet(output_path)

print("Saved processed dataset to:", output_path)

spark.stop()