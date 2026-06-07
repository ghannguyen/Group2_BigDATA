from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, year, month, weekofyear, sum as spark_sum
import os
import glob
import shutil

spark = SparkSession.builder \
    .appName("Walmart_Preprocess") \
    .getOrCreate()

# 1. Read raw CSV files
# nullValue="NA" giúp Spark hiểu NA là giá trị thiếu
train = spark.read.csv(
    "data/raw/train.csv",
    header=True,
    inferSchema=True,
    nullValue="NA"
)

features = spark.read.csv(
    "data/raw/features.csv",
    header=True,
    inferSchema=True,
    nullValue="NA"
)

stores = spark.read.csv(
    "data/raw/stores.csv",
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

# 7. Save as one CSV file
output_dir = "data/processed/walmart_sales_enriched"
final_csv_name = "walmart_sales_enriched.csv"

df.coalesce(1) \
  .write \
  .mode("overwrite") \
  .option("header", True) \
  .csv(output_dir)

# 8. Rename Spark part file to walmart_sales_enriched.csv
part_file = glob.glob(os.path.join(output_dir, "part-*.csv"))[0]
final_path = os.path.join(output_dir, final_csv_name)

if os.path.exists(final_path):
    os.remove(final_path)

shutil.move(part_file, final_path)

# 9. Remove unnecessary Spark/Hadoop output files
for file_path in glob.glob(os.path.join(output_dir, "_SUCCESS")):
    os.remove(file_path)

for file_path in glob.glob(os.path.join(output_dir, ".*.crc")):
    os.remove(file_path)

print("Saved final CSV to:", final_path)

spark.stop()