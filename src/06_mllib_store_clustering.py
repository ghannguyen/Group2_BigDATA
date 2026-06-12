from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Imputer,
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# ============================================================
# SPARK MLLIB EXTENSION - STORE CLUSTERING BY KMEANS
# Dataset: Walmart Store Sales
# Input: processed Parquet on HDFS
# Output: store-level features, clustered results, KMeans model on HDFS
# ============================================================

spark = (
    SparkSession.builder
    .appName("Nhom_02_Walmart_MLlib_Store_Clustering")
    .config("spark.ui.port", "4050")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.adaptive.enabled", "false")
    .getOrCreate()
)

# Giam log Spark de ket qua terminal de nhin hon
spark.sparkContext.setLogLevel("ERROR")

# Tat ca du lieu dau vao/dau ra deu nam tren HDFS, khong doc file CSV local
processed_path = "hdfs://localhost:9000/bigdata/walmart/processed/walmart_sales_enriched"
store_feature_path = "hdfs://localhost:9000/bigdata/walmart/processed/store_level_features"
prepared_feature_path = "hdfs://localhost:9000/bigdata/walmart/processed/store_clustering_prepared"
cluster_result_path = "hdfs://localhost:9000/bigdata/walmart/processed/store_cluster_results"
preprocess_model_path = "hdfs://localhost:9000/bigdata/walmart/models/kmeans_store_preprocess_pipeline"
kmeans_model_path = "hdfs://localhost:9000/bigdata/walmart/models/kmeans_store_clustering"

print("=" * 100)
print("SPARK MLLIB EXTENSION - STORE CLUSTERING BY KMEANS")
print("=" * 100)


# ============================================================
# PART 1 - STORE LEVEL DATA PREPARATION - Member 1
# ============================================================

print("=" * 100)
print("PART 1 - READ PROCESSED DATA FROM HDFS")
print("=" * 100)

# Doc bang processed da join train.csv, features.csv, stores.csv tu HDFS
df = spark.read.parquet(processed_path)
df.createOrReplaceTempView("walmart_sales_enriched")

print("Processed path:", processed_path)
print("Rows:", df.count())
print("Columns:", len(df.columns))
df.printSchema()

# Persist bang processed vi PART 1 se tinh nhieu chi so tong hop tu bang nay
df.persist(StorageLevel.MEMORY_AND_DISK)
df.count()

print("=" * 100)
print("PART 1 - CREATE STORE LEVEL FEATURES")
print("=" * 100)

# KMeans phan cum theo Store, nen can dua du lieu ve cap cua hang.
# Moi dong trong bang store_level_features dai dien cho 1 cua hang Walmart.
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
            COALESCE(MarkDown1, 0) AS MarkDown1,
            COALESCE(MarkDown2, 0) AS MarkDown2,
            COALESCE(MarkDown3, 0) AS MarkDown3,
            COALESCE(MarkDown4, 0) AS MarkDown4,
            COALESCE(MarkDown5, 0) AS MarkDown5,
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

    store_week_sales AS (
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
        FROM store_week_sales
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
        SUM(CASE WHEN markdown_record_ratio IS NULL THEN 1 ELSE 0 END) AS markdown_record_ratio_null,
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


# ============================================================
# PART 2 - KMEANS PREPROCESSING PIPELINE - Member 2
# ============================================================

print("=" * 100)
print("PART 2 - KMEANS PREPROCESSING PIPELINE")
print("=" * 100)

# Doc lai store-level features tu HDFS de chung minh PART 2 nhan output cua PART 1
store_df = spark.read.parquet(store_feature_path)
store_df.createOrReplaceTempView("store_level_features_from_hdfs")

print("Store feature path:", store_feature_path)
print("Store feature rows:", store_df.count())
print("Store feature columns:", len(store_df.columns))

# Cac cot so dung cho KMeans.
# KMeans dua tren khoang cach, nen can StandardScaler de chuan hoa scale.
numeric_cols = [
    "Size",
    "total_records",
    "total_departments",
    "total_weeks",
    "total_sales",
    "avg_weekly_sales",
    "min_weekly_sales",
    "max_weekly_sales",
    "avg_store_week_sales",
    "std_store_week_sales",
    "coefficient_variation",
    "avg_markdown",
    "markdown_record_ratio",
    "holiday_sales_ratio",
    "avg_temperature",
    "avg_fuel_price",
    "avg_cpi",
    "avg_unemployment"
]

imputer = Imputer(
    inputCols=numeric_cols,
    outputCols=[c + "_imp" for c in numeric_cols]
).setStrategy("median")

type_indexer = StringIndexer(
    inputCol="Type",
    outputCol="TypeIndex",
    handleInvalid="keep"
)

type_encoder = OneHotEncoder(
    inputCols=["TypeIndex"],
    outputCols=["TypeVec"]
)

assembler = VectorAssembler(
    inputCols=[c + "_imp" for c in numeric_cols] + ["TypeVec"],
    outputCol="raw_features"
)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withStd=True,
    withMean=True
)

preprocessing_pipeline = Pipeline(stages=[
    imputer,
    type_indexer,
    type_encoder,
    assembler,
    scaler
])

preprocessing_model = preprocessing_pipeline.fit(store_df)
prepared_data = preprocessing_model.transform(store_df)

print("=" * 100)
print("PREPARED DATA SAMPLE FOR KMEANS")
print("=" * 100)

prepared_data.select(
    "Store",
    "Type",
    "Size",
    "total_sales",
    "avg_weekly_sales",
    "coefficient_variation",
    "avg_markdown",
    "holiday_sales_ratio",
    "features"
).show(20, truncate=False)

print("Prepared data rows:", prepared_data.count())
print("Prepared data columns:", len(prepared_data.columns))

print("=" * 100)
print("SAVE PREPARED FEATURES AND PREPROCESSING MODEL TO HDFS")
print("=" * 100)

prepared_data.write.mode("overwrite").parquet(prepared_feature_path)
preprocessing_model.write().overwrite().save(preprocess_model_path)

print("Prepared features saved to:", prepared_feature_path)
print("Preprocessing model saved to:", preprocess_model_path)


# ============================================================
# PART 3 - KMEANS TRAINING, EVALUATION AND INTERPRETATION - Member 3
# ============================================================
print("=" * 100)
print("PART 3 - TRAIN KMEANS STORE CLUSTERING MODEL")
print("=" * 100)

# Doc lai prepared data tu HDFS de chung minh PART 3 nhan output cua PART 2
kmeans_data = spark.read.parquet(prepared_feature_path)

print("Prepared feature path:", prepared_feature_path)
print("Prepared data rows:", kmeans_data.count())
print("Prepared data columns:", len(kmeans_data.columns))

# k = 4 de chia store thanh 4 nhom van hanh
kmeans = KMeans(
    featuresCol="features",
    predictionCol="cluster",
    k=4,
    seed=42
)

kmeans_model = kmeans.fit(kmeans_data)
clustered = kmeans_model.transform(kmeans_data)
clustered.createOrReplaceTempView("store_cluster_results")

print("KMeans training completed.")

print("=" * 100)
print("STORE CLUSTERING SAMPLE")
print("=" * 100)

clustered.select(
    "Store",
    "Type",
    "Size",
    "total_sales",
    "avg_weekly_sales",
    "coefficient_variation",
    "avg_markdown",
    "holiday_sales_ratio",
    "cluster"
).orderBy("cluster", "Store").show(45, truncate=False)

print("=" * 100)
print("KMEANS EVALUATION")
print("=" * 100)

evaluator = ClusteringEvaluator(
    featuresCol="features",
    predictionCol="cluster",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

silhouette = evaluator.evaluate(clustered)
print("Silhouette Score:", round(silhouette, 4))

print("=" * 100)
print("CLUSTER SUMMARY FOR BUSINESS INTERPRETATION")
print("=" * 100)

cluster_summary = spark.sql("""
    SELECT
        cluster,
        COUNT(*) AS total_stores,
        ROUND(AVG(Size), 2) AS avg_size,
        ROUND(AVG(total_sales), 2) AS avg_total_sales,
        ROUND(AVG(avg_weekly_sales), 2) AS avg_weekly_sales,
        ROUND(AVG(coefficient_variation), 4) AS avg_coefficient_variation,
        ROUND(AVG(avg_markdown), 2) AS avg_markdown,
        ROUND(AVG(markdown_record_ratio), 4) AS avg_markdown_record_ratio,
        ROUND(AVG(holiday_sales_ratio), 4) AS avg_holiday_sales_ratio,
        ROUND(AVG(avg_cpi), 3) AS avg_cpi,
        ROUND(AVG(avg_unemployment), 3) AS avg_unemployment
    FROM store_cluster_results
    GROUP BY cluster
    ORDER BY cluster
""")

cluster_summary.show(truncate=False)

print("=" * 100)
print("SAVE CLUSTER RESULTS AND MODEL TO HDFS")
print("=" * 100)

clustered.coalesce(1).write.mode("overwrite").parquet(cluster_result_path)
kmeans_model.write().overwrite().save(kmeans_model_path)

print("Cluster results saved to:", cluster_result_path)
print("KMeans model saved to:", kmeans_model_path)

print("=" * 100)
print("DONE - STORE CLUSTERING EXTENSION COMPLETED")
print("=" * 100)

# Giai phong cache neu PART 1 co persist df
try:
    df.unpersist()
except Exception:
    pass

input("Nhan Enter de dung Spark...")
spark.stop()
