from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel

spark = (
    SparkSession.builder
    .appName("Nhom_02_Walmart_MLlib_Store_Clustering")
    .config("spark.ui.port", "4050")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.adaptive.enabled", "false")
    .getOrCreate()
)

# Giam log de output de nhin hon
spark.sparkContext.setLogLevel("ERROR")

# Du lieu processed da duoc tao tu train.csv, features.csv va stores.csv
# Duong dan nay doc truc tiep tu HDFS, khong doc local
processed_path = "hdfs://localhost:9000/bigdata/walmart/processed/walmart_sales_enriched"

# Output cua PART 1: bang tong hop theo Store de dung cho KMeans
store_feature_path = "hdfs://localhost:9000/bigdata/walmart/processed/store_level_features"

print("=" * 100)
print("SPARK MLLIB EXTENSION - STORE CLUSTERING BY KMEANS")
print("=" * 100)

# ============================================================
# PART 1 - STORE LEVEL DATA PREPARATION - Member 1
# ============================================================

print("=" * 100)
print("PART 1 - READ PROCESSED DATA FROM HDFS")
print("=" * 100)

df = spark.read.parquet(processed_path)
df.createOrReplaceTempView("walmart_sales_enriched")

print("Processed path:", processed_path)
print("Rows:", df.count())
print("Columns:", len(df.columns))
df.printSchema()

# Persist vi bang processed duoc dung de tong hop nhieu chi so store-level
df.persist(StorageLevel.MEMORY_AND_DISK)
df.count()

print("=" * 100)
print("PART 1 - CREATE STORE LEVEL FEATURES")
print("=" * 100)

# Muc tieu:
# KMeans khong nen chay tren tung dong weekly sales,
# ma nen chay tren bang da tong hop moi Store la 1 dong.
# Bang nay mo ta quy mo, doanh so, do bien dong, markdown va dieu kien kinh te cua tung cua hang.

store_features = spark.sql("""
    WITH base AS (
        SELECT
            Store,
            Type,
            Size,
            Date,
            Dept,
            Weekly_Sales,
            IsHoliday,
            Temperature,
            Fuel_Price,
            MarkDown1,
            MarkDown2,
            MarkDown3,
            MarkDown4,
            MarkDown5,
            CPI,
            Unemployment,
            COALESCE(MarkDown1, 0)
            + COALESCE(MarkDown2, 0)
            + COALESCE(MarkDown3, 0)
            + COALESCE(MarkDown4, 0)
            + COALESCE(MarkDown5, 0) AS markdown_total
        FROM walmart_sales_enriched
        WHERE Weekly_Sales IS NOT NULL
          AND Weekly_Sales >= 0
    ),
    store_week AS (
        SELECT
            Store,
            Type,
            Size,
            Date,
            SUM(Weekly_Sales) AS store_week_sales
        FROM base
        GROUP BY Store, Type, Size, Date
    ),
    store_volatility AS (
        SELECT
            Store,
            ROUND(AVG(store_week_sales), 2) AS avg_store_week_sales,
            ROUND(STDDEV(store_week_sales), 2) AS std_store_week_sales,
            ROUND(
                STDDEV(store_week_sales) / AVG(store_week_sales),
                4
            ) AS coefficient_variation
        FROM store_week
        GROUP BY Store
    ),
    store_summary AS (
        SELECT
            Store,
            Type,
            Size,
            COUNT(*) AS total_records,
            COUNT(DISTINCT Dept) AS total_departments,
            COUNT(DISTINCT Date) AS total_weeks,
            ROUND(SUM(Weekly_Sales), 2) AS total_sales,
            ROUND(AVG(Weekly_Sales), 2) AS avg_weekly_sales,
            ROUND(MIN(Weekly_Sales), 2) AS min_weekly_sales,
            ROUND(MAX(Weekly_Sales), 2) AS max_weekly_sales,
            ROUND(AVG(markdown_total), 2) AS avg_markdown,
            ROUND(
                SUM(CASE WHEN markdown_total > 0 THEN 1 ELSE 0 END) / COUNT(*),
                4
            ) AS markdown_record_ratio,
            ROUND(
                SUM(CASE WHEN IsHoliday = true THEN Weekly_Sales ELSE 0 END) / SUM(Weekly_Sales),
                4
            ) AS holiday_sales_ratio,
            ROUND(AVG(Temperature), 2) AS avg_temperature,
            ROUND(AVG(Fuel_Price), 3) AS avg_fuel_price,
            ROUND(AVG(CPI), 3) AS avg_cpi,
            ROUND(AVG(Unemployment), 3) AS avg_unemployment
        FROM base
        GROUP BY Store, Type, Size
    )
    SELECT
        s.Store,
        s.Type,
        s.Size,
        s.total_records,
        s.total_departments,
        s.total_weeks,
        s.total_sales,
        s.avg_weekly_sales,
        s.min_weekly_sales,
        s.max_weekly_sales,
        v.avg_store_week_sales,
        v.std_store_week_sales,
        v.coefficient_variation,
        s.avg_markdown,
        s.markdown_record_ratio,
        s.holiday_sales_ratio,
        s.avg_temperature,
        s.avg_fuel_price,
        s.avg_cpi,
        s.avg_unemployment
    FROM store_summary s
    LEFT JOIN store_volatility v
        ON s.Store = v.Store
    ORDER BY s.Store
""")

store_features.createOrReplaceTempView("store_level_features")

print("Store-level rows:", store_features.count())
print("Store-level columns:", len(store_features.columns))
store_features.printSchema()

print("=" * 100)
print("STORE LEVEL FEATURE SAMPLE")
print("=" * 100)

store_features.show(20, truncate=False)

print("=" * 100)
print("CHECK MISSING VALUES IN STORE LEVEL FEATURES")
print("=" * 100)

spark.sql("""
    SELECT
        SUM(CASE WHEN Store IS NULL THEN 1 ELSE 0 END) AS Store_null,
        SUM(CASE WHEN Type IS NULL THEN 1 ELSE 0 END) AS Type_null,
        SUM(CASE WHEN Size IS NULL THEN 1 ELSE 0 END) AS Size_null,
        SUM(CASE WHEN total_sales IS NULL THEN 1 ELSE 0 END) AS total_sales_null,
        SUM(CASE WHEN avg_weekly_sales IS NULL THEN 1 ELSE 0 END) AS avg_weekly_sales_null,
        SUM(CASE WHEN std_store_week_sales IS NULL THEN 1 ELSE 0 END) AS std_store_week_sales_null,
        SUM(CASE WHEN coefficient_variation IS NULL THEN 1 ELSE 0 END) AS coefficient_variation_null,
        SUM(CASE WHEN avg_markdown IS NULL THEN 1 ELSE 0 END) AS avg_markdown_null,
        SUM(CASE WHEN holiday_sales_ratio IS NULL THEN 1 ELSE 0 END) AS holiday_sales_ratio_null,
        SUM(CASE WHEN avg_temperature IS NULL THEN 1 ELSE 0 END) AS avg_temperature_null,
        SUM(CASE WHEN avg_fuel_price IS NULL THEN 1 ELSE 0 END) AS avg_fuel_price_null,
        SUM(CASE WHEN avg_cpi IS NULL THEN 1 ELSE 0 END) AS avg_cpi_null,
        SUM(CASE WHEN avg_unemployment IS NULL THEN 1 ELSE 0 END) AS avg_unemployment_null
    FROM store_level_features
""").show(truncate=False)

print("=" * 100)
print("SAVE STORE LEVEL FEATURES TO HDFS")
print("=" * 100)

store_features.write.mode("overwrite").parquet(store_feature_path)

print("Store-level features saved to:", store_feature_path)

print("=" * 100)
print("PART 1 COMPLETED")
print("Nguoi 2 se doc store_level_features de lam Pipeline cho KMeans.")
print("=" * 100)

df.unpersist()

input("Nhan Enter de dung Spark...")
spark.stop()