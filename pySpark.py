"""
FULL PYSPARK ETL PIPELINE (DETAILED, SINGLE FILE)

Covers:
1. Spark Session creation
2. Extract (CSV / Parquet / JDBC)
3. Schema enforcement
4. Data cleaning & validation
5. Transformations & enrichment
6. Aggregations & analytics
7. Window functions
8. EDA (statistics & profiling)
9. Data quality checks
10. Performance optimizations
11. Load (Parquet / JDBC)
"""

# 1. IMPORTS

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType,
    DoubleType, TimestampType
)

from pyspark.sql.functions import (
    col, trim, lower, upper,
    to_timestamp, to_date,
    when, count, sum as spark_sum,
    avg, max as spark_max, min as spark_min,
    desc, expr, broadcast,
    approx_count_distinct
)

from pyspark.sql.window import Window


# 2. SPARK SESSION CREATION

def create_spark_session():
    """
    Creates and returns a SparkSession.
    In production, configs depend on cluster size & workload.
    """
    spark = (
        SparkSession.builder
        .appName("Full_ETL_PySpark")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    return spark



# 3. SCHEMA DEFINITIONS (IMPORTANT FOR PERFORMANCE)

sales_schema = StructType([
    StructField("order_id", StringType(), False),
    StructField("user_id", StringType(), True),
    StructField("product_id", StringType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("price", DoubleType(), True),
    StructField("order_timestamp", StringType(), True),
    StructField("country", StringType(), True)
])

products_schema = StructType([
    StructField("product_id", StringType(), False),
    StructField("product_name", StringType(), True),
    StructField("category", StringType(), True),
    StructField("launch_date", StringType(), True)
])



# 4. EXTRACT PHASE

def extract_data(spark):
    """
    Extract data from multiple sources.
    """

    # ---- CSV source ----
    sales_df = (
        spark.read
        .option("header", True)
        .schema(sales_schema)
        .csv("/data/sales.csv")
    )

    # ---- Parquet source ----
    products_df = (
        spark.read
        .schema(products_schema)
        .parquet("/data/products.parquet")
    )

    return sales_df, products_df


# 5. DATA CLEANING & STANDARDIZATION

def clean_sales_data(df):
    """
    Cleans raw sales data:
    - trims strings
    - fixes data types
    - handles missing values
    """

    df = (
        df
        .withColumn("country", upper(trim(col("country"))))
        .withColumn("quantity", col("quantity").cast("int"))
        .withColumn("price", col("price").cast("double"))
    )

    # Convert timestamp string → timestamp type
    df = df.withColumn(
        "order_ts",
        to_timestamp(col("order_timestamp"), "yyyy-MM-dd HH:mm:ss")
    )

    # Drop rows with critical missing values
    df = df.dropna(subset=["order_id", "product_id"])

    # Fill defaults
    df = df.fillna({
        "quantity": 0,
        "country": "UNKNOWN"
    })

    # Remove duplicates
    df = df.dropDuplicates(["order_id"])

    return df


def clean_products_data(df):
    """
    Cleans products dimension data.
    """
    df = (
        df
        .withColumn("product_name", trim(col("product_name")))
        .withColumn("category", lower(trim(col("category"))))
        .withColumn("launch_date", to_date(col("launch_date")))
    )

    return df



# 6. TRANSFORM & ENRICH

def enrich_sales(sales_df, products_df):
    """
    Join sales with products and create derived columns.
    """

    # Broadcast small dimension table
    enriched_df = sales_df.join(
        broadcast(products_df),
        on="product_id",
        how="left"
    )

    # Derived metrics
    enriched_df = (
        enriched_df
        .withColumn("total_amount", col("quantity") * col("price"))
        .withColumn(
            "price_bucket",
            when(col("price") < 50, "low")
            .when(col("price") < 200, "medium")
            .otherwise("high")
        )
    )

    return enriched_df



# 7. ANALYTICS & AGGREGATIONS

def sales_analytics(df):
    """
    Common business aggregations.
    """

    # Revenue by country
    revenue_by_country = (
        df.groupBy("country")
        .agg(
            spark_sum("total_amount").alias("revenue"),
            count("*").alias("orders")
        )
        .orderBy(desc("revenue"))
    )

    # Revenue by category
    revenue_by_category = (
        df.groupBy("category")
        .agg(spark_sum("total_amount").alias("revenue"))
        .orderBy(desc("revenue"))
    )

    return revenue_by_country, revenue_by_category


# 8. WINDOW FUNCTIONS (VERY IMPORTANT FOR INTERVIEWS)

def window_metrics(df):
    """
    Ranking & running metrics using window functions.
    """

    # Rank products by revenue within each country
    window_country = Window.partitionBy("country").orderBy(desc("total_amount"))

    ranked_df = df.withColumn(
        "rank_in_country",
        expr("row_number() over (partition by country order by total_amount desc)")
    )

    return ranked_df



# 9. EDA (EXPLORATORY DATA ANALYSIS)
def eda_profile(df):
    """
    Basic exploratory data analysis.
    """

    print("Row count:", df.count())
    df.printSchema()

    # Summary statistics
    df.describe(["quantity", "price", "total_amount"]).show()

    # Cardinality
    df.select(
        approx_count_distinct("user_id").alias("unique_users"),
        approx_count_distinct("product_id").alias("unique_products")
    ).show()

    # Missing value analysis
    missing = df.select([
        count(when(col(c).isNull(), c)).alias(c)
        for c in df.columns
    ])
    missing.show()



# 10. DATA QUALITY CHECKS

def data_quality_checks(df):
    """
    Validates correctness of data.
    """

    # Negative values
    invalid_rows = df.filter(
        (col("quantity") < 0) | (col("price") < 0)
    )

    print("Invalid rows count:", invalid_rows.count())

    # Orphan records (missing product info)
    orphan_products = df.filter(col("product_name").isNull())
    print("Missing product mappings:", orphan_products.count())



# 11. LOAD PHASE
def load_data(df):
    """
    Write curated data to storage.
    """

    # Parquet for analytics
    (
        df.write
        .mode("overwrite")
        .partitionBy("country", "category")
        .parquet("/curated/sales_enriched")
    )

    # Example JDBC load
    """
    df.write.format("jdbc") \
        .option("url", "jdbc:postgresql://host/db") \
        .option("dbtable", "analytics.sales") \
        .option("user", "user") \
        .option("password", "password") \
        .mode("append") \
        .save()
    """



# 12. MAIN DRIVER
def main():
    spark = create_spark_session()

    sales_df, products_df = extract_data(spark)

    sales_clean = clean_sales_data(sales_df)
    products_clean = clean_products_data(products_df)

    enriched_df = enrich_sales(sales_clean, products_clean)

    # Cache for reuse
    enriched_df.cache()

    # Analytics
    revenue_country, revenue_category = sales_analytics(enriched_df)
    revenue_country.show()
    revenue_category.show()

    # Window metrics
    ranked_df = window_metrics(enriched_df)
    ranked_df.show(5)

    # EDA
    eda_profile(enriched_df)

    # Data quality
    data_quality_checks(enriched_df)

    # Load
    load_data(enriched_df)

    spark.stop()



# 13. ENTRY POINT
if __name__ == "__main__":
    main()