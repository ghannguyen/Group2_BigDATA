from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel

# Khoi tao SparkSession de chay Spark SQL
spark = SparkSession.builder \
    .appName("Nhom_02_Walmart_Big_Data") \
    .config("spark.ui.port", "4050") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.sql.adaptive.enabled", "false") \
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


# Cau 1: Phan khuc cua hang theo doanh so, quy mo va hieu suat tren dien tich
# Muc dich: khong chi xem cua hang nao ban nhieu, ma con xem cua hang nao ban hieu qua so voi quy mo.
query_1 = spark.sql("""
    WITH store_sales AS (
        SELECT
            Store,
            Type,
            Size,
            ROUND(SUM(Weekly_Sales), 2) AS total_sales,
            ROUND(AVG(Weekly_Sales), 2) AS avg_weekly_sales,
            COUNT(DISTINCT Dept) AS total_departments,
            ROUND(SUM(Weekly_Sales) / Size, 4) AS sales_per_size
        FROM walmart_sales_enriched
        GROUP BY Store, Type, Size
    ),
    store_segment AS (
        SELECT
            Store,
            Type,
            Size,
            total_sales,
            avg_weekly_sales,
            total_departments,
            sales_per_size,
            CASE
                WHEN Size < 100000 THEN 'Small Store'
                WHEN Size BETWEEN 100000 AND 180000 THEN 'Medium Store'
                ELSE 'Large Store'
            END AS size_group,
            NTILE(4) OVER (
                PARTITION BY Type
                ORDER BY total_sales DESC
            ) AS sales_quartile_in_type,
            DENSE_RANK() OVER (
                PARTITION BY Type
                ORDER BY sales_per_size DESC
            ) AS efficiency_rank_in_type
        FROM store_sales
    )
    SELECT
        Store,
        Type,
        size_group,
        Size,
        total_departments,
        total_sales,
        avg_weekly_sales,
        sales_per_size,
        sales_quartile_in_type,
        efficiency_rank_in_type
    FROM store_segment
    WHERE sales_quartile_in_type = 1
       OR efficiency_rank_in_type <= 5
    ORDER BY Type, efficiency_rank_in_type, total_sales DESC
""")

print("\n" + "=" * 90)
print("CAU 1: Phan khuc cua hang theo doanh so, quy mo va hieu suat")
print("=" * 90)
query_1.show(80, truncate=False)


# Cau 2: Doanh so on dinh hay bien dong theo tung loai cua hang
# Muc dich: tim cac cua hang co doanh so bien dong manh qua cac tuan de ho tro quan ly ton kho.
query_2 = spark.sql("""
    WITH store_week_sales AS (
        SELECT
            Store,
            Type,
            YEAR(Date) AS sales_year,
            WEEKOFYEAR(Date) AS week_of_year,
            ROUND(SUM(Weekly_Sales), 2) AS store_week_sales
        FROM walmart_sales_enriched
        GROUP BY Store, Type, YEAR(Date), WEEKOFYEAR(Date)
    ),
    store_volatility AS (
        SELECT
            Store,
            Type,
            ROUND(AVG(store_week_sales), 2) AS avg_store_week_sales,
            ROUND(STDDEV(store_week_sales), 2) AS std_store_week_sales,
            ROUND(STDDEV(store_week_sales) / AVG(store_week_sales), 4) AS coefficient_variation,
            COUNT(*) AS total_weeks
        FROM store_week_sales
        GROUP BY Store, Type
        HAVING COUNT(*) >= 30
    ),
    ranked_volatility AS (
        SELECT
            Store,
            Type,
            avg_store_week_sales,
            std_store_week_sales,
            coefficient_variation,
            total_weeks,
            DENSE_RANK() OVER (
                PARTITION BY Type
                ORDER BY coefficient_variation DESC
            ) AS volatility_rank
        FROM store_volatility
    )
    SELECT
        Store,
        Type,
        avg_store_week_sales,
        std_store_week_sales,
        coefficient_variation,
        total_weeks,
        volatility_rank
    FROM ranked_volatility
    WHERE volatility_rank <= 5
    ORDER BY Type, volatility_rank
""")

print("\n" + "=" * 90)
print("CAU 2: Top cua hang co doanh so bien dong manh theo tung Type")
print("=" * 90)
query_2.show(50, truncate=False)


# Cau 3: Hieu qua Markdown theo Type va nhom quy mo cua hang
# Muc dich: so sanh doanh so trung binh khi co khuyen mai va khi khong co khuyen mai.
query_3 = spark.sql("""
    WITH markdown_base AS (
        SELECT
            Store,
            Type,
            Size,
            Weekly_Sales,
            COALESCE(MarkDown1, 0)
            + COALESCE(MarkDown2, 0)
            + COALESCE(MarkDown3, 0)
            + COALESCE(MarkDown4, 0)
            + COALESCE(MarkDown5, 0) AS markdown_total,
            CASE
                WHEN Size < 100000 THEN 'Small Store'
                WHEN Size BETWEEN 100000 AND 180000 THEN 'Medium Store'
                ELSE 'Large Store'
            END AS size_group
        FROM walmart_sales_enriched
    ),
    markdown_compare AS (
        SELECT
            Type,
            size_group,
            COUNT(*) AS total_records,
            ROUND(AVG(CASE WHEN markdown_total > 0 THEN Weekly_Sales END), 2) AS avg_sales_with_markdown,
            ROUND(AVG(CASE WHEN markdown_total = 0 THEN Weekly_Sales END), 2) AS avg_sales_without_markdown,
            ROUND(AVG(CASE WHEN markdown_total > 0 THEN markdown_total END), 2) AS avg_markdown_when_used,
            SUM(CASE WHEN markdown_total > 0 THEN 1 ELSE 0 END) AS records_with_markdown,
            SUM(CASE WHEN markdown_total = 0 THEN 1 ELSE 0 END) AS records_without_markdown
        FROM markdown_base
        GROUP BY Type, size_group
    )
    SELECT
        Type,
        size_group,
        total_records,
        records_with_markdown,
        records_without_markdown,
        avg_markdown_when_used,
        avg_sales_with_markdown,
        avg_sales_without_markdown,
        ROUND(avg_sales_with_markdown - avg_sales_without_markdown, 2) AS markdown_sales_lift,
        ROUND(
            ((avg_sales_with_markdown - avg_sales_without_markdown) / avg_sales_without_markdown) * 100,
            2
        ) AS markdown_lift_percent
    FROM markdown_compare
    WHERE avg_sales_with_markdown IS NOT NULL
      AND avg_sales_without_markdown IS NOT NULL
      AND avg_sales_without_markdown > 0
    ORDER BY markdown_lift_percent DESC
""")

print("\n" + "=" * 90)
print("CAU 3: So sanh doanh so co Markdown va khong Markdown theo Type va quy mo")
print("=" * 90)
query_3.show(50, truncate=False)


# Cau 4: Tac dong cua dieu kien kinh te den doanh so so voi trung binh cua tung Type
# Muc dich: xem doanh so thay doi the nao trong cac boi canh Fuel Price, CPI va Unemployment khac nhau.
query_4 = spark.sql("""
    WITH economic_group AS (
        SELECT
            Type,
            CASE
                WHEN Fuel_Price < 3 THEN 'Low Fuel Price'
                WHEN Fuel_Price BETWEEN 3 AND 3.7 THEN 'Medium Fuel Price'
                ELSE 'High Fuel Price'
            END AS fuel_group,
            CASE
                WHEN CPI < 150 THEN 'Low CPI'
                WHEN CPI BETWEEN 150 AND 220 THEN 'Medium CPI'
                ELSE 'High CPI'
            END AS cpi_group,
            CASE
                WHEN Unemployment < 6 THEN 'Low Unemployment'
                WHEN Unemployment BETWEEN 6 AND 9 THEN 'Medium Unemployment'
                ELSE 'High Unemployment'
            END AS unemployment_group,
            Weekly_Sales
        FROM walmart_sales_enriched
        WHERE Fuel_Price IS NOT NULL
          AND CPI IS NOT NULL
          AND Unemployment IS NOT NULL
    ),
    condition_sales AS (
        SELECT
            Type,
            fuel_group,
            cpi_group,
            unemployment_group,
            COUNT(*) AS total_records,
            ROUND(SUM(Weekly_Sales), 2) AS total_sales,
            ROUND(AVG(Weekly_Sales), 2) AS avg_weekly_sales
        FROM economic_group
        GROUP BY Type, fuel_group, cpi_group, unemployment_group
    ),
    type_benchmark AS (
        SELECT
            Type,
            ROUND(AVG(Weekly_Sales), 2) AS avg_sales_by_type
        FROM walmart_sales_enriched
        GROUP BY Type
    )
    SELECT
        c.Type,
        c.fuel_group,
        c.cpi_group,
        c.unemployment_group,
        c.total_records,
        c.total_sales,
        c.avg_weekly_sales,
        b.avg_sales_by_type,
        ROUND(c.avg_weekly_sales - b.avg_sales_by_type, 2) AS difference_from_type_avg,
        ROUND(
            ((c.avg_weekly_sales - b.avg_sales_by_type) / b.avg_sales_by_type) * 100,
            2
        ) AS difference_percent
    FROM condition_sales c
    JOIN type_benchmark b
        ON c.Type = b.Type
    WHERE c.total_records >= 1000
    ORDER BY difference_percent DESC
""")

print("\n" + "=" * 90)
print("CAU 4: Doanh so theo dieu kien kinh te va so sanh voi trung binh Type")
print("=" * 90)
query_4.show(80, truncate=False)

print("\n" + "=" * 90)
print("EXPLAIN CHO CAU 4")
print("=" * 90)
query_4.explain(True)
# Cau 5: Top 5 cua hang tang truong doanh so manh nhat giua cac nam
# Muc dich: xac dinh cac cua hang co toc do tang truong doanh so cao nhat so voi nam truoc.
query_5 = spark.sql("""
    WITH yearly_store_sales AS (
        SELECT
            Store,
            Type,
            Year,
            ROUND(SUM(Weekly_Sales), 2) AS yearly_sales
        FROM walmart_sales_enriched
        GROUP BY Store, Type, Year
    ),
    growth_table AS (
        SELECT
            Store,
            Type,
            Year,
            yearly_sales,
            LAG(yearly_sales) OVER (
                PARTITION BY Store
                ORDER BY Year
            ) AS previous_year_sales
        FROM yearly_store_sales
    )
    SELECT
        Store,
        Type,
        Year,
        yearly_sales,
        previous_year_sales,
        ROUND(yearly_sales - previous_year_sales, 2) AS sales_change,
        ROUND(((yearly_sales - previous_year_sales) / previous_year_sales) * 100, 2) AS growth_rate_percent
    FROM growth_table
    WHERE previous_year_sales IS NOT NULL
    ORDER BY growth_rate_percent DESC
    LIMIT 5
""")


print("\n" + "=" * 90)
print("CAU 5: Top 5 cua hang tang truong doanh so manh nhat giua cac nam")
print("=" * 90)
query_5.show(50, truncate=False)




# Cau 6: Phan tich department co doanh so cao hon trung binh cua chinh cua hang
# Muc dich: tim cac department dong gop noi bat va vuot muc trung binh doanh thu trong tung cua hang.
query_6 = spark.sql("""
    WITH dept_store_sales AS (
        SELECT
            Store,
            Type,
            Dept,
            ROUND(SUM(Weekly_Sales), 2) AS dept_total_sales
        FROM walmart_sales_enriched
        GROUP BY Store, Type, Dept
    ),
    store_avg AS (
        SELECT
            Store,
            ROUND(AVG(dept_total_sales), 2) AS avg_dept_sales_in_store
        FROM dept_store_sales
        GROUP BY Store
    )
    SELECT
        d.Store,
        d.Type,
        d.Dept,
        d.dept_total_sales,
        s.avg_dept_sales_in_store,
        ROUND(d.dept_total_sales - s.avg_dept_sales_in_store, 2) AS difference_from_store_avg
    FROM dept_store_sales d
    JOIN store_avg s
        ON d.Store = s.Store
    WHERE d.dept_total_sales > s.avg_dept_sales_in_store
    ORDER BY difference_from_store_avg DESC
    LIMIT 10
""")


print("\n" + "=" * 90)
print("CAU 6: Department co doanh so cao hon trung binh cua chinh cua hang")
print("=" * 90)
query_6.show(50, truncate=False)




# Cau 7: Xep hang nhom nhiet do co doanh so cao nhat theo tung loai cua hang
# Muc dich: phan tich dieu kien nhiet do nao tao ra doanh so trung binh cao nhat trong tung Type.
query_7 = spark.sql("""
    WITH temperature_sales AS (
        SELECT
            Type,
            CASE
                WHEN Temperature < 40 THEN 'Cold'
                WHEN Temperature BETWEEN 40 AND 70 THEN 'Mild'
                ELSE 'Hot'
            END AS temperature_group,
            COUNT(*) AS total_records,
            ROUND(AVG(Temperature), 2) AS avg_temperature,
            ROUND(SUM(Weekly_Sales), 2) AS total_sales,
            ROUND(AVG(Weekly_Sales), 2) AS avg_weekly_sales
        FROM walmart_sales_enriched
        GROUP BY
            Type,
            CASE
                WHEN Temperature < 40 THEN 'Cold'
                WHEN Temperature BETWEEN 40 AND 70 THEN 'Mild'
                ELSE 'Hot'
            END
    ),
    ranked_temperature AS (
        SELECT
            Type,
            temperature_group,
            total_records,
            avg_temperature,
            total_sales,
            avg_weekly_sales,
            DENSE_RANK() OVER (
                PARTITION BY Type
                ORDER BY avg_weekly_sales DESC
            ) AS temperature_rank
        FROM temperature_sales
    )
    SELECT
        Type,
        temperature_group,
        total_records,
        avg_temperature,
        total_sales,
        avg_weekly_sales,
        temperature_rank
    FROM ranked_temperature
    ORDER BY Type, temperature_rank
""")


print("\n" + "=" * 90)
print("CAU 7: Xep hang nhom nhiet do co doanh so cao nhat theo tung loai cua hang")
print("=" * 90)
query_7.show(50, truncate=False)




# Cau 8: Xep hang tuan co doanh so cao nhat trong tung nam
# Muc dich: tim cac tuan cao diem doanh so cua tung nam de ho tro lap ke hoach ton kho va khuyen mai.
query_8 = spark.sql("""
    WITH weekly_sales AS (
        SELECT
            Year,
            WeekOfYear,
            IsHoliday,
            ROUND(SUM(Weekly_Sales), 2) AS total_weekly_sales,
            ROUND(AVG(Weekly_Sales), 2) AS avg_weekly_sales,
            COUNT(DISTINCT Store) AS total_stores
        FROM walmart_sales_enriched
        GROUP BY Year, WeekOfYear, IsHoliday
    ),
    ranked_weeks AS (
        SELECT
            Year,
            WeekOfYear,
            IsHoliday,
            total_weekly_sales,
            avg_weekly_sales,
            total_stores,
            DENSE_RANK() OVER (
                PARTITION BY Year
                ORDER BY total_weekly_sales DESC
            ) AS week_rank
        FROM weekly_sales
    )
    SELECT
        Year,
        WeekOfYear,
        IsHoliday,
        total_weekly_sales,
        avg_weekly_sales,
        total_stores,
        week_rank
    FROM ranked_weeks
    WHERE week_rank <= 3
    ORDER BY Year, week_rank
""")


print("\n" + "=" * 90)
print("CAU 8: Top 3 tuan co doanh so cao nhat trong tung nam")
print("=" * 90)
query_8.show(50, truncate=False)

print("\n" + "=" * 90)
print("EXPLAIN CHO CAU 8")
print("=" * 90)
query_8.explain(True)


# Cau 9: Ty trong doanh thu cua tung Department trong tong doanh thu cua Store
# Muc dich: xac dinh nhom nganh hang nao dong gop nhieu nhat cho tung cua hang.
query_9 = spark.sql("""
    WITH dept_store_sales AS (
        SELECT
            Store,
            Type,
            Dept,
            ROUND(SUM(Weekly_Sales), 2) AS dept_sales
        FROM walmart_sales_enriched
        GROUP BY Store, Type, Dept
    ),
    store_total_sales AS (
        SELECT
            Store,
            Type,
            Dept,
            dept_sales,
            SUM(dept_sales) OVER (
                PARTITION BY Store
            ) AS store_total_sales
        FROM dept_store_sales
    ),
    ranked_dept AS (
        SELECT
            Store,
            Type,
            Dept,
            dept_sales,
            store_total_sales,
            ROUND((dept_sales / store_total_sales) * 100, 2) AS contribution_percent,
            DENSE_RANK() OVER (
                PARTITION BY Store
                ORDER BY dept_sales DESC
            ) AS dept_rank
        FROM store_total_sales
    )
    SELECT
        Store,
        Type,
        Dept,
        dept_sales,
        store_total_sales,
        contribution_percent,
        dept_rank
    FROM ranked_dept
    WHERE dept_rank <= 3
    ORDER BY Store, dept_rank
""")

print("\n" + "=" * 90)
print("CAU 9: Ty trong doanh thu cua tung Department trong tong doanh thu cua Store")
print("=" * 90)
query_9.show(100, truncate=False)


# Cau 10: Phat hien Store-Dept co doanh so bat thuong so voi trung binh Department
# Muc dich: tim cac cua hang co doanh so cua mot Dept cao/thap bat thuong so voi mat bang chung.
query_10 = spark.sql("""
    WITH store_dept_sales AS (
        SELECT
            Store,
            Type,
            Dept,
            ROUND(SUM(Weekly_Sales), 2) AS store_dept_sales
        FROM walmart_sales_enriched
        GROUP BY Store, Type, Dept
    ),
    dept_benchmark AS (
        SELECT
            Store,
            Type,
            Dept,
            store_dept_sales,
            ROUND(
                AVG(store_dept_sales) OVER (
                    PARTITION BY Dept
                ),
                2
            ) AS avg_dept_sales_all_stores
        FROM store_dept_sales
    )
    SELECT
        Store,
        Type,
        Dept,
        store_dept_sales,
        avg_dept_sales_all_stores,
        ROUND(store_dept_sales - avg_dept_sales_all_stores, 2) AS sales_gap,
        ROUND(
            ((store_dept_sales - avg_dept_sales_all_stores) / avg_dept_sales_all_stores) * 100,
            2
        ) AS gap_percent
    FROM dept_benchmark
    WHERE avg_dept_sales_all_stores > 0
    ORDER BY ABS(gap_percent) DESC
    LIMIT 20
""")

print("\n" + "=" * 90)
print("CAU 10: Phat hien Store-Dept co doanh so bat thuong so voi trung binh Department")
print("=" * 90)
query_10.show(50, truncate=False)


# Cau 11: Moving Average doanh so theo thang
# Muc dich: tinh trung binh truot 3 thang de phan tich xu huong doanh so on dinh hon.
query_11 = spark.sql("""
    WITH monthly_sales AS (
        SELECT
            year,
            month,
            ROUND(SUM(Weekly_Sales), 2) AS monthly_sales,
            ROUND(AVG(Weekly_Sales), 2) AS avg_weekly_sales,
            COUNT(*) AS total_records
        FROM walmart_sales_enriched
        GROUP BY year, month
    )
    SELECT
        year,
        month,
        monthly_sales,
        avg_weekly_sales,
        total_records,
        ROUND(
            AVG(monthly_sales) OVER (
                ORDER BY year, month
                ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
            ),
            2
        ) AS moving_avg_3_months
    FROM monthly_sales
    ORDER BY year, month
""")

print("\n" + "=" * 90)
print("CAU 11: Moving Average doanh so theo thang")
print("=" * 90)
query_11.show(50, truncate=False)


# Cau 12: Phan tich doanh so da chieu bang CUBE
# Muc dich: tong hop doanh so theo Type, nam va nhom ngay le o nhieu cap do khac nhau.
query_12 = spark.sql("""
    SELECT
        COALESCE(Type, 'ALL_TYPES') AS Type,
        COALESCE(CAST(year AS STRING), 'ALL_YEARS') AS sales_year,
        COALESCE(holiday_group, 'ALL_HOLIDAY_GROUPS') AS holiday_group,
        COUNT(*) AS total_records,
        COUNT(DISTINCT Store) AS total_stores,
        ROUND(SUM(Weekly_Sales), 2) AS total_sales,
        ROUND(AVG(Weekly_Sales), 2) AS avg_weekly_sales
    FROM (
        SELECT
            Type,
            year,
            CASE
                WHEN IsHoliday = true THEN 'Holiday Week'
                ELSE 'Normal Week'
            END AS holiday_group,
            Store,
            Weekly_Sales
        FROM walmart_sales_enriched
    ) t
    GROUP BY CUBE(Type, year, holiday_group)
    ORDER BY Type, sales_year, holiday_group
""")

print("\n" + "=" * 90)
print("CAU 12: Phan tich doanh so da chieu bang CUBE")
print("=" * 90)
query_12.show(100, truncate=False)

print("\n" + "=" * 90)
print("EXPLAIN CHO CAU 12")
print("=" * 90)
query_12.explain(True)
# Giai phong cache sau khi chay xong
df.unpersist()
input("Nhan Enter de dung Spark...")
spark.stop()
