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
# SPARK MLLIB EXTENSION - STORE OPERATIONAL CLUSTERING BY KMEANS
# Dataset: Walmart Store Sales
# Input : processed Parquet on HDFS
# Output: store-level features, clustered results, KMeans model on HDFS
# ============================================================

spark = (
    SparkSession.builder
    .appName("Nhom_02_Walmart_MLlib_Store_Operational_Clustering")
    .config("spark.ui.port", "4050")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.adaptive.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# Tat ca du lieu dau vao/dau ra deu nam tren HDFS, khong doc file local
processed_path = "hdfs://localhost:9000/bigdata/walmart/processed/walmart_sales_enriched"

store_feature_path = "hdfs://localhost:9000/bigdata/walmart/processed/store_operational_features"
prepared_feature_path = "hdfs://localhost:9000/bigdata/walmart/processed/store_operational_clustering_prepared"
cluster_result_path = "hdfs://localhost:9000/bigdata/walmart/processed/store_operational_cluster_results"

preprocess_model_path = "hdfs://localhost:9000/bigdata/walmart/models/kmeans_store_operational_preprocess_pipeline"
kmeans_model_path = "hdfs://localhost:9000/bigdata/walmart/models/kmeans_store_operational_clustering"


print("=" * 100)
print("SPARK MLLIB EXTENSION - STORE OPERATIONAL CLUSTERING BY KMEANS")
print("=" * 100)


# ============================================================
# PART 1 - STORE OPERATIONAL FEATURE ENGINEERING - Member 1
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

# Persist vi bang processed duoc dung nhieu lan de tong hop feature cap Store
df.persist(StorageLevel.MEMORY_AND_DISK)
df.count()

print("=" * 100)
print("PART 1 - CREATE STORE OPERATIONAL FEATURES")
print("=" * 100)

# Muc tieu cua bai toan:
# Phan cum cua hang Walmart theo dac diem van hanh.
# Moi dong trong bang store_operational_features dai dien cho 1 Store.
#
# Bo feature rut gon gom 4 nhom y nghia:
# 1. Quy mo: Size, total_departments
# 2. Hieu qua kinh doanh: avg_store_week_sales, sales_per_size
# 3. Do on dinh doanh so: coefficient_variation
# 4. Khuyen mai va mua vu: avg_markdown, holiday_sales_ratio
#
# Cac feature nay ngan gon hon ban dau nhung van du y nghia kinh doanh:
# quy mo - hieu qua - bien dong - khuyen mai/ngay le.

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
            COALESCE(MarkDown1, 0) AS MarkDown1,
            COALESCE(MarkDown2, 0) AS MarkDown2,
            COALESCE(MarkDown3, 0) AS MarkDown3,
            COALESCE(MarkDown4, 0) AS MarkDown4,
            COALESCE(MarkDown5, 0) AS MarkDown5,
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
            COUNT(DISTINCT Dept) AS total_departments,

            ROUND(SUM(Weekly_Sales), 2) AS total_sales,
            ROUND(SUM(Weekly_Sales) / Size, 4) AS sales_per_size,

            ROUND(AVG(markdown_total), 2) AS avg_markdown,
            ROUND(
                SUM(CASE WHEN IsHoliday = true THEN Weekly_Sales ELSE 0 END) / SUM(Weekly_Sales),
                4
            ) AS holiday_sales_ratio

        FROM base
        GROUP BY Store, Type, Size
    )

    SELECT
        s.Store,
        s.Type,
        s.Size,
        s.total_departments,
        s.total_sales,
        v.avg_store_week_sales,
        s.sales_per_size,
        v.coefficient_variation,
        s.avg_markdown,
        s.holiday_sales_ratio
    FROM store_summary s
    LEFT JOIN store_volatility v
        ON s.Store = v.Store
    ORDER BY s.Store
""")

store_features.createOrReplaceTempView("store_operational_features")

print("Store-level rows:", store_features.count())
print("Store-level columns:", len(store_features.columns))
store_features.printSchema()

print("=" * 100)
print("STORE OPERATIONAL FEATURE SAMPLE")
print("=" * 100)

store_features.show(45, truncate=False)

print("=" * 100)
print("CHECK MISSING VALUES IN STORE OPERATIONAL FEATURES")
print("=" * 100)

spark.sql("""
    SELECT
        SUM(CASE WHEN Store IS NULL THEN 1 ELSE 0 END) AS Store_null,
        SUM(CASE WHEN Type IS NULL THEN 1 ELSE 0 END) AS Type_null,
        SUM(CASE WHEN Size IS NULL THEN 1 ELSE 0 END) AS Size_null,
        SUM(CASE WHEN total_departments IS NULL THEN 1 ELSE 0 END) AS total_departments_null,
        SUM(CASE WHEN total_sales IS NULL THEN 1 ELSE 0 END) AS total_sales_null,
        SUM(CASE WHEN avg_store_week_sales IS NULL THEN 1 ELSE 0 END) AS avg_store_week_sales_null,
        SUM(CASE WHEN sales_per_size IS NULL THEN 1 ELSE 0 END) AS sales_per_size_null,
        SUM(CASE WHEN coefficient_variation IS NULL THEN 1 ELSE 0 END) AS coefficient_variation_null,
        SUM(CASE WHEN avg_markdown IS NULL THEN 1 ELSE 0 END) AS avg_markdown_null,
        SUM(CASE WHEN holiday_sales_ratio IS NULL THEN 1 ELSE 0 END) AS holiday_sales_ratio_null
    FROM store_operational_features
""").show(truncate=False)

print("=" * 100)
print("SAVE STORE OPERATIONAL FEATURES TO HDFS")
print("=" * 100)

store_features.write.mode("overwrite").parquet(store_feature_path)

print("Store operational features saved to:", store_feature_path)


# ============================================================
# PART 2 - KMEANS PREPROCESSING PIPELINE - Member 2
# ============================================================

print("=" * 100)
print("PART 2 - KMEANS PREPROCESSING PIPELINE")
print("=" * 100)

# Doc lai output PART 1 tu HDFS de chung minh flow lam viec tuan tu
store_df = spark.read.parquet(store_feature_path)
store_df.createOrReplaceTempView("store_operational_features_from_hdfs")

print("Store feature path:", store_feature_path)
print("Store feature rows:", store_df.count())
print("Store feature columns:", len(store_df.columns))
store_df.printSchema()

# Bo cot so rut gon va co y nghia kinh doanh ro rang cho KMeans
numeric_cols = [
    "Size",
    "total_departments",
    "avg_store_week_sales",
    "sales_per_size",
    "coefficient_variation",
    "avg_markdown",
    "holiday_sales_ratio"
]

# Xu ly missing values bang median
imputer = Imputer(
    inputCols=numeric_cols,
    outputCols=[c + "_imp" for c in numeric_cols]
).setStrategy("median")

# Ma hoa Type A/B/C
type_indexer = StringIndexer(
    inputCol="Type",
    outputCol="TypeIndex",
    handleInvalid="keep"
)

type_encoder = OneHotEncoder(
    inputCols=["TypeIndex"],
    outputCols=["TypeVec"]
)

# Gom feature so va TypeVec thanh raw_features
assembler = VectorAssembler(
    inputCols=[c + "_imp" for c in numeric_cols] + ["TypeVec"],
    outputCol="raw_features"
)

# KMeans dua tren khoang cach nen can StandardScaler
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
    "total_departments",
    "avg_store_week_sales",
    "sales_per_size",
    "coefficient_variation",
    "avg_markdown",
    "holiday_sales_ratio",
).show(45, truncate=False)

print("Prepared data rows:", prepared_data.count())
print("Prepared data columns:", len(prepared_data.columns))

print("=" * 100)
print("SAVE PREPARED FEATURES AND PREPROCESSING MODEL TO HDFS")
print("=" * 100)

prepared_data.write.mode("overwrite").parquet(prepared_feature_path)
preprocessing_model.write().overwrite().save(preprocess_model_path)

print("Prepared features saved to:", prepared_feature_path)
print("Preprocessing model saved to:", preprocess_model_path)