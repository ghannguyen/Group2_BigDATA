from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel

# Khoi tao SparkSession de chay Spark SQL
spark = SparkSession.builder \
    .appName("Nhom_02_Walmart_Big_Data") \
    .config("spark.ui.port", "4050") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

# Giam log cua Spark de ket qua truy van de nhin hon
spark.sparkContext.setLogLevel("ERROR")
# Duong dan den du lieu da xu ly tren HDFS
# Day la file Parquet duoc tao tu train.csv, features.csv va stores.csv
processed_path = "hdfs://localhost:9000/bigdata/walmart/processed/walmart_sales_enriched"

# Doc du lieu truc tiep tu HDFS, khong doc tu may local
df = spark.read.parquet(processed_path)

# Tao bang tam de viet truy van bang Spark SQL
df.createOrReplaceTempView("walmart_sales_enriched")

print("=" * 90)
print("KIEM TRA DU LIEU DA DOC TU HDFS")
print("=" * 90)
print("HDFS path:", processed_path)
print("So dong:", df.count())
print("So cot:", len(df.columns))
df.printSchema()

# Bang nay se duoc dung nhieu lan cho cac cau truy van,
# nen persist de han che viec Spark doc lai du lieu tu HDFS nhieu lan.
df.persist(StorageLevel.MEMORY_AND_DISK)
df.count()

print("Da persist bang walmart_sales_enriched thanh cong.")


# Cau 1: Tong doanh so theo loai cua hang va nam
# Muc dich: xem loai cua hang nao tao ra doanh so cao hon qua tung nam.
query_1 = spark.sql("""
    SELECT
        Type,
        YEAR(Date) AS sales_year,
        COUNT(*) AS total_records,
        COUNT(DISTINCT Store) AS total_stores,
        ROUND(SUM(Weekly_Sales), 2) AS total_sales,
        ROUND(AVG(Weekly_Sales), 2) AS avg_weekly_sales
    FROM walmart_sales_enriched
    GROUP BY Type, YEAR(Date)
    ORDER BY sales_year, total_sales DESC
""")

print("\n" + "=" * 90)
print("CAU 1: Tong doanh so theo loai cua hang va nam")
print("=" * 90)
query_1.show(50, truncate=False)


# Cau 2: Top department co doanh so cao nhat trong tung loai cua hang
# Muc dich: tim cac nhom nganh hang chu luc cua tung loai cua hang A/B/C.
query_2 = spark.sql("""
    WITH dept_sales AS (
        SELECT
            Type,
            Dept,
            ROUND(SUM(Weekly_Sales), 2) AS total_sales,
            ROUND(AVG(Weekly_Sales), 2) AS avg_sales,
            COUNT(*) AS total_records
        FROM walmart_sales_enriched
        GROUP BY Type, Dept
    ),
    ranked_dept AS (
        SELECT
            Type,
            Dept,
            total_sales,
            avg_sales,
            total_records,
            DENSE_RANK() OVER (
                PARTITION BY Type
                ORDER BY total_sales DESC
            ) AS dept_rank
        FROM dept_sales
    )
    SELECT
        Type,
        Dept,
        total_sales,
        avg_sales,
        total_records,
        dept_rank
    FROM ranked_dept
    WHERE dept_rank <= 3
    ORDER BY Type, dept_rank
""")

print("\n" + "=" * 90)
print("CAU 2: Top 3 department co doanh so cao nhat trong tung loai cua hang")
print("=" * 90)
query_2.show(50, truncate=False)


# Cau 3: Phan tich doanh so theo ngay le va khuyen mai
# Muc dich: xem doanh so thay doi nhu the nao khi co holiday va markdown.
query_3 = spark.sql("""
    WITH markdown_table AS (
        SELECT
            *,
            COALESCE(MarkDown1, 0)
            + COALESCE(MarkDown2, 0)
            + COALESCE(MarkDown3, 0)
            + COALESCE(MarkDown4, 0)
            + COALESCE(MarkDown5, 0) AS markdown_total
        FROM walmart_sales_enriched
    )
    SELECT
        CASE
            WHEN IsHoliday = true THEN 'Holiday Week'
            ELSE 'Normal Week'
        END AS week_type,
        CASE
            WHEN markdown_total = 0 THEN 'No Markdown'
            WHEN markdown_total < 5000 THEN 'Low Markdown'
            WHEN markdown_total BETWEEN 5000 AND 20000 THEN 'Medium Markdown'
            ELSE 'High Markdown'
        END AS markdown_group,
        COUNT(*) AS total_records,
        ROUND(AVG(markdown_total), 2) AS avg_markdown,
        ROUND(SUM(Weekly_Sales), 2) AS total_sales,
        ROUND(AVG(Weekly_Sales), 2) AS avg_weekly_sales
    FROM markdown_table
    GROUP BY
        CASE
            WHEN IsHoliday = true THEN 'Holiday Week'
            ELSE 'Normal Week'
        END,
        CASE
            WHEN markdown_total = 0 THEN 'No Markdown'
            WHEN markdown_total < 5000 THEN 'Low Markdown'
            WHEN markdown_total BETWEEN 5000 AND 20000 THEN 'Medium Markdown'
            ELSE 'High Markdown'
        END
    ORDER BY week_type, avg_weekly_sales DESC
""")

print("\n" + "=" * 90)
print("CAU 3: Anh huong cua Holiday va Markdown den doanh so")
print("=" * 90)
query_3.show(50, truncate=False)


# Cau 4: Tang truong doanh so theo thang
# Muc dich: so sanh doanh so tung thang voi thang truoc do bang LAG().
query_4 = spark.sql("""
    WITH monthly_sales AS (
        SELECT
            YEAR(Date) AS sales_year,
            MONTH(Date) AS sales_month,
            ROUND(SUM(Weekly_Sales), 2) AS monthly_sales,
            ROUND(AVG(Weekly_Sales), 2) AS avg_weekly_sales,
            COUNT(*) AS total_records
        FROM walmart_sales_enriched
        GROUP BY YEAR(Date), MONTH(Date)
    ),
    monthly_growth AS (
        SELECT
            sales_year,
            sales_month,
            monthly_sales,
            avg_weekly_sales,
            total_records,
            LAG(monthly_sales) OVER (
                ORDER BY sales_year, sales_month
            ) AS previous_month_sales
        FROM monthly_sales
    )
    SELECT
        sales_year,
        sales_month,
        monthly_sales,
        previous_month_sales,
        ROUND(monthly_sales - previous_month_sales, 2) AS sales_change,
        ROUND(
            ((monthly_sales - previous_month_sales) / previous_month_sales) * 100,
            2
        ) AS growth_rate_percent,
        avg_weekly_sales,
        total_records
    FROM monthly_growth
    ORDER BY sales_year, sales_month
""")

print("\n" + "=" * 90)
print("CAU 4: Tang truong doanh so theo thang")
print("=" * 90)
query_4.show(50, truncate=False)

# In execution plan de co minh chung cho phan explain()
# Phan nay co the dua vao diem cong toi uu hieu nang.
print("\n" + "=" * 90)
print("EXPLAIN CHO CAU 4")
print("=" * 90)
query_4.explain(True)

# Giai phong cache sau khi chay xong
df.unpersist()

input("Nhan Enter de dung Spark...")
spark.stop()