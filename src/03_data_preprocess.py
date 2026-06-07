from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, year, month, weekofyear

spark = SparkSession.builder \
    .appName("Walmart_Preprocess") \
    .getOrCreate()

train = spark.read.csv("data/raw/train.csv", header=True, inferSchema=True)
features = spark.read.csv("data/raw/features.csv", header=True, inferSchema=True)
stores = spark.read.csv("data/raw/stores.csv", header=True, inferSchema=True)

df = train.join(features, ["Store", "Date", "IsHoliday"], "left") \
          .join(stores, ["Store"], "left")

df = df.fillna({
    "MarkDown1": 0,
    "MarkDown2": 0,
    "MarkDown3": 0,
    "MarkDown4": 0,
    "MarkDown5": 0,
    "CPI": 0,
    "Unemployment": 0
})

df = df.withColumn("Date", to_date(col("Date"), "yyyy-MM-dd")) \
       .withColumn("Year", year(col("Date"))) \
       .withColumn("Month", month(col("Date"))) \
       .withColumn("WeekOfYear", weekofyear(col("Date")))

df.write.mode("overwrite").parquet("data/processed/walmart_sales_enriched")

print("Saved to data/processed/walmart_sales_enriched")
print("Rows:", df.count())
print("Columns:", len(df.columns))
df.printSchema()

spark.stop()