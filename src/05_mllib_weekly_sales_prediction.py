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