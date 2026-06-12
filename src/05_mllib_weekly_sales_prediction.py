from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Imputer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
# PART 3 - MODEL TRAINING AND EVALUATION - Member 3
# ============================================================

# RandomForestRegressor duoc dung de du bao Weekly_Sales.
# Mo hinh nay phu hop vi doanh so ban le co the bi anh huong boi nhieu yeu to
# nhu cua hang, department, ngay le, mua vu, thoi tiet va khuyen mai.
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="Weekly_Sales",
    predictionCol="prediction",
    numTrees=20,
    maxDepth=6,
    maxBins=32,
    subsamplingRate=0.7,
    seed=42
)

# Lay cac stage tien xu ly tu PART 2.
# Mot so file co the dat ten la type_indexer/type_encoder,
# mot so file co the dat ten ngan la indexer/encoder.
indexer_stage = type_indexer if "type_indexer" in globals() else indexer
encoder_stage = type_encoder if "type_encoder" in globals() else encoder

# Gom pipeline tien xu ly va model vao mot quy trinh chung.
pipeline = Pipeline(stages=[
    imputer,
    indexer_stage,
    encoder_stage,
    assembler,
    rf
])

print("=" * 100)
print("PART 3 - TRAIN RANDOM FOREST REGRESSION MODEL")
print("=" * 100)

# Train model tren tap train da chia o PART 2
model = pipeline.fit(train_df)

print("Model training completed.")

print("=" * 100)
print("PREDICTION SAMPLE ON TEST DATA")
print("=" * 100)

# Du doan tren tap test
pred = model.transform(test_df)

# In mau ket qua du doan de dua vao bao cao
pred.select(
    "Store",
    "Dept",
    "Type",
    "sales_year",
    "sales_month",
    "week_of_year",
    "Weekly_Sales",
    "prediction"
).show(30, truncate=False)

print("=" * 100)
print("MODEL EVALUATION")
print("=" * 100)

# Danh gia mo hinh bang RegressionEvaluator
rmse = RegressionEvaluator(
    labelCol="Weekly_Sales",
    predictionCol="prediction",
    metricName="rmse"
).evaluate(pred)

mae = RegressionEvaluator(
    labelCol="Weekly_Sales",
    predictionCol="prediction",
    metricName="mae"
).evaluate(pred)

r2 = RegressionEvaluator(
    labelCol="Weekly_Sales",
    predictionCol="prediction",
    metricName="r2"
).evaluate(pred)

print("RMSE:", round(rmse, 4))
print("MAE :", round(mae, 4))
print("R2  :", round(r2, 4))
print("=" * 100)
print("=" * 100)
print("VISUALIZATION - ACTUAL VS PREDICTED WEEKLY SALES")
print("=" * 100)

# Lay mot phan du lieu test de ve bieu do, tranh collect qua nhieu dong ve local
plot_df = (
    pred
    .select("Weekly_Sales", "prediction")
    .dropna()
    .sample(withReplacement=False, fraction=0.08, seed=42)
    .limit(5000)
    .toPandas()
)

print("Visualization sample rows:", len(plot_df))

# Tao folder luu anh local de dua vao bao cao
output_dir = "screenshots/04_mllib"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------------------------
# Figure 1: Full actual vs predicted chart
# ------------------------------------------------------------
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=plot_df,
    x="Weekly_Sales",
    y="prediction",
    alpha=0.4
)

min_value = min(plot_df["Weekly_Sales"].min(), plot_df["prediction"].min())
max_value = max(plot_df["Weekly_Sales"].max(), plot_df["prediction"].max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    color="red",
    linestyle="--",
    label="Perfect Prediction"
)

plt.title("Actual vs Predicted Weekly Sales - Random Forest")
plt.xlabel("Actual Weekly Sales")
plt.ylabel("Predicted Weekly Sales")
plt.legend()
plt.tight_layout()

output_chart_path = f"{output_dir}/rf_actual_vs_predicted_weekly_sales.png"
plt.savefig(output_chart_path, dpi=300)
plt.close()

print("Visualization saved to:", output_chart_path)

# ------------------------------------------------------------
# Figure 2: Zoomed chart for main sales range
# ------------------------------------------------------------
plot_df_zoom = plot_df[plot_df["Weekly_Sales"] <= 100000]

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=plot_df_zoom,
    x="Weekly_Sales",
    y="prediction",
    alpha=0.4
)

min_value = min(plot_df_zoom["Weekly_Sales"].min(), plot_df_zoom["prediction"].min())
max_value = max(plot_df_zoom["Weekly_Sales"].max(), plot_df_zoom["prediction"].max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    color="red",
    linestyle="--",
    label="Perfect Prediction"
)

plt.title("Actual vs Predicted Weekly Sales - Random Forest (Zoomed)")
plt.xlabel("Actual Weekly Sales")
plt.ylabel("Predicted Weekly Sales")
plt.legend()
plt.tight_layout()

output_zoom_path = f"{output_dir}/rf_actual_vs_predicted_weekly_sales_zoomed.png"
plt.savefig(output_zoom_path, dpi=300)
plt.close()

print("Zoomed visualization saved to:", output_zoom_path)
print("=" * 100)
print("SAVE MODEL TO HDFS")
print("=" * 100)

# Luu model len HDFS de chung minh model khong chi chay local
model.write().overwrite().save(model_output_path)

print("Model saved to:", model_output_path)

print("=" * 100)
print("DONE - PART 3 COMPLETED")
print("=" * 100)

plt.close()

print("Zoomed visualization saved to:", output_zoom_path)

input("Nhan Enter de dung Spark...")
spark.stop()