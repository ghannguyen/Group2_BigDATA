from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Walmart_Spark_SQL_Queries") \
    .getOrCreate()

df = spark.read.csv(
    "data/processed/walmart_sales_enriched.csv",
    header=True,
    inferSchema=True
)

df.createOrReplaceTempView("walmart_sales")

queries = {
    "Q1_total_sales_by_year": """
        SELECT Year,
               ROUND(SUM(Weekly_Sales), 2) AS total_sales
        FROM walmart_sales
        GROUP BY Year
        ORDER BY Year
    """,

    "Q2_monthly_sales": """
        SELECT Year, Month,
               ROUND(SUM(Weekly_Sales), 2) AS monthly_sales
        FROM walmart_sales
        GROUP BY Year, Month
        ORDER BY Year, Month
    """,

    "Q3_top_10_stores": """
        SELECT Store,
               ROUND(SUM(Weekly_Sales), 2) AS total_sales
        FROM walmart_sales
        GROUP BY Store
        ORDER BY total_sales DESC
        LIMIT 10
    """,

    "Q4_sales_by_store_type": """
        SELECT Type,
               COUNT(DISTINCT Store) AS number_of_stores,
               ROUND(SUM(Weekly_Sales), 2) AS total_sales,
               ROUND(AVG(Weekly_Sales), 2) AS avg_sales
        FROM walmart_sales
        GROUP BY Type
        ORDER BY total_sales DESC
    """,

    "Q5_holiday_vs_nonholiday": """
        SELECT IsHoliday,
               ROUND(SUM(Weekly_Sales), 2) AS total_sales,
               ROUND(AVG(Weekly_Sales), 2) AS avg_sales,
               COUNT(*) AS record_count
        FROM walmart_sales
        GROUP BY IsHoliday
        ORDER BY IsHoliday DESC
    """,

    "Q6_top_departments": """
        SELECT Dept,
               ROUND(SUM(Weekly_Sales), 2) AS total_sales,
               ROUND(AVG(Weekly_Sales), 2) AS avg_sales
        FROM walmart_sales
        GROUP BY Dept
        ORDER BY total_sales DESC
        LIMIT 10
    """,

    "Q7_store_rank_by_year": """
        SELECT Year, Store, total_sales,
               RANK() OVER(PARTITION BY Year ORDER BY total_sales DESC) AS sales_rank
        FROM (
            SELECT Year, Store,
                   ROUND(SUM(Weekly_Sales), 2) AS total_sales
            FROM walmart_sales
            GROUP BY Year, Store
        ) t
        ORDER BY Year, sales_rank
        LIMIT 30
    """,

    "Q8_above_average_stores": """
        SELECT Store,
               ROUND(SUM(Weekly_Sales), 2) AS total_sales
        FROM walmart_sales
        GROUP BY Store
        HAVING SUM(Weekly_Sales) > (
            SELECT AVG(store_sales)
            FROM (
                SELECT Store, SUM(Weekly_Sales) AS store_sales
                FROM walmart_sales
                GROUP BY Store
            ) x
        )
        ORDER BY total_sales DESC
    """,

    "Q9_promotion_effect": """
        SELECT CASE
                   WHEN (MarkDown1 + MarkDown2 + MarkDown3 + MarkDown4 + MarkDown5) > 0
                   THEN 'Has Promotion'
                   ELSE 'No Promotion'
               END AS promotion_status,
               ROUND(SUM(Weekly_Sales), 2) AS total_sales,
               ROUND(AVG(Weekly_Sales), 2) AS avg_sales,
               COUNT(*) AS record_count
        FROM walmart_sales
        GROUP BY CASE
                   WHEN (MarkDown1 + MarkDown2 + MarkDown3 + MarkDown4 + MarkDown5) > 0
                   THEN 'Has Promotion'
                   ELSE 'No Promotion'
                 END
    """,

    "Q10_weekly_sales_change": """
        SELECT Store, Dept, Date, Weekly_Sales,
               LAG(Weekly_Sales) OVER(PARTITION BY Store, Dept ORDER BY Date) AS previous_week_sales,
               ROUND(
                   Weekly_Sales - LAG(Weekly_Sales) OVER(PARTITION BY Store, Dept ORDER BY Date), 
                   2
               ) AS sales_change
        FROM walmart_sales
        ORDER BY Store, Dept, Date
        LIMIT 50
    """
}

for name, query in queries.items():
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    result = spark.sql(query)
    result.show(30, truncate=False)

spark.stop()