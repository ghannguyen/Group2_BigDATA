from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Imputer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
spark = (
SparkSession.builder
.appName("Nhom_02_Walmart_MLlib_Weekly_Sales")
.config("spark.ui.port", "4050")
.config("spark.sql.shuffle.partitions", "8")
.config("spark.sql.adaptive.enabled", "false")
.getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
processed_path = "hdfs://localhost:9000/bigdata/walmart/processed/walmart_sales_enriched"
model_output_path = "hdfs://localhost:9000/bigdata/walmart/models/rf_weekly_sales_prediction"
print("=" * 100)
print("SPARK MLLIB - WEEKLY SALES PREDICTION")
print("=" * 100)
# ============================================================
# PART 1 - DATA PREPARATION - Member 1
# ============================================================
df = spark.read.parquet(processed_path)
df.createOrReplaceTempView("walmart_sales_enriched")
print("Processed path:", processed_path)
print("Rows:", df.count())
print("Columns:", len(df.columns))
df.printSchema()
ml_data = spark.sql("""
SELECT
Store,
Dept,
Type,
Size,
IsHoliday,
Year AS sales_year,
Month AS sales_month,
WeekOfYear AS week_of_year,
Temperature,
Fuel_Price,
MarkDown1,
MarkDown2,
MarkDown3,
MarkDown4,
MarkDown5,
CPI,
Unemployment,
Weekly_Sales
FROM walmart_sales_enriched
WHERE Weekly_Sales IS NOT NULL
AND Weekly_Sales >= 0
""")
ml_data.createOrReplaceTempView("ml_data")
print("ML data rows:", ml_data.count())
print("ML data columns:", len(ml_data.columns))
ml_data.show(10, truncate=False)
print("=" * 100)
print("CHECK MISSING VALUES BEFORE PIPELINE")
print("=" * 100)
spark.sql("""
SELECT
SUM(CASE WHEN Store IS NULL THEN 1 ELSE 0 END) AS Store_null,
SUM(CASE WHEN Dept IS NULL THEN 1 ELSE 0 END) AS Dept_null,
SUM(CASE WHEN Type IS NULL THEN 1 ELSE 0 END) AS Type_null,
SUM(CASE WHEN Size IS NULL THEN 1 ELSE 0 END) AS Size_null,
SUM(CASE WHEN Temperature IS NULL THEN 1 ELSE 0 END) AS Temperature_null,
SUM(CASE WHEN Fuel_Price IS NULL THEN 1 ELSE 0 END) AS Fuel_Price_null,
SUM(CASE WHEN MarkDown1 IS NULL THEN 1 ELSE 0 END) AS MarkDown1_null,
SUM(CASE WHEN MarkDown2 IS NULL THEN 1 ELSE 0 END) AS MarkDown2_null,
SUM(CASE WHEN MarkDown3 IS NULL THEN 1 ELSE 0 END) AS MarkDown3_null,
SUM(CASE WHEN MarkDown4 IS NULL THEN 1 ELSE 0 END) AS MarkDown4_null,
SUM(CASE WHEN MarkDown5 IS NULL THEN 1 ELSE 0 END) AS MarkDown5_null,
SUM(CASE WHEN CPI IS NULL THEN 1 ELSE 0 END) AS CPI_null,
SUM(CASE WHEN Unemployment IS NULL THEN 1 ELSE 0 END) AS Unemployment_null,
SUM(CASE WHEN Weekly_Sales IS NULL THEN 1 ELSE 0 END) AS Weekly_Sales_null
FROM ml_data
""").show(truncate=False)
# ============================================================

# ============================================================
# PART 2 - MLLIB PIPELINE PREPROCESSING - Member 2
# ============================================================

# Chuyen IsHoliday tu boolean thanh numeric 1/0 de mo hinh MLlib co the su dung
ml_data = ml_data.withColumn(
    "IsHolidayNum",
    when(col("IsHoliday") == True, 1.0).otherwise(0.0)
)

# Cac cot numeric duoc su dung lam feature dau vao
numeric_cols = [
    "Store",
    "Dept",
    "Size",
    "IsHolidayNum",
    "sales_year",
    "sales_month",
    "week_of_year",
    "Temperature",
    "Fuel_Price",
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5",
    "CPI",
    "Unemployment"
]

# Xu ly missing values cho cac cot numeric bang median
imputer = Imputer(
    inputCols=numeric_cols,
    outputCols=[c + "_imp" for c in numeric_cols]
).setStrategy("median")

# Ma hoa cot Type A/B/C thanh chi so numeric
type_indexer = StringIndexer(
    inputCol="Type",
    outputCol="TypeIndex",
    handleInvalid="keep"
)

# Chuyen TypeIndex thanh vector nhi phan TypeVec
type_encoder = OneHotEncoder(
    inputCols=["TypeIndex"],
    outputCols=["TypeVec"]
)

# Gom tat ca feature da xu ly thanh mot cot features
assembler = VectorAssembler(
    inputCols=[c + "_imp" for c in numeric_cols] + ["TypeVec"],
    outputCol="features"
)

# Chia du lieu thanh train/test theo ti le 80/20
train_df, test_df = ml_data.randomSplit([0.8, 0.2], seed=42)

print("=" * 100)
print("SPLIT TRAIN / TEST")
print("=" * 100)
print("Train rows:", train_df.count())
print("Test rows:", test_df.count())
# ============================================================