# spark_processor.py - Humanized Spark Data Processor
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pandas as pd
import logging

class SparkDataProcessor:
    """
    Big Data processor for invoice datasets using Apache Spark.
    Handles batch analytics, real-time streams, and fallback processing.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.spark = None
        self.initialize_spark()

    # -----------------------------
    # Spark Initialization
    # -----------------------------
    def initialize_spark(self):
        """Initialize Spark session with optimized settings."""
        try:
            self.spark = SparkSession.builder \
                .appName("InvoicePaymentAnalytics") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.sql.adaptive.skew.enabled", "true") \
                .config("spark.executor.memory", "2g") \
                .config("spark.driver.memory", "1g") \
                .getOrCreate()

            self.spark.sparkContext.setLogLevel("ERROR")
            self.logger.info("✅ Spark session initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Spark: {e}")
            self.spark = None

    # -----------------------------
    # Dataset Processing
    # -----------------------------
    def process_large_dataset(self, pandas_df: pd.DataFrame, use_spark_threshold: int = 100000) -> pd.DataFrame:
        """
        Process datasets using Spark if size exceeds threshold.
        Falls back to pandas for smaller datasets or if Spark is unavailable.
        """
        if self.spark is None or len(pandas_df) < use_spark_threshold:
            self.logger.warning("Spark not used, falling back to pandas processing")
            return self._pandas_fallback_processing(pandas_df)

        try:
            spark_df = self.spark.createDataFrame(pandas_df)
            enriched_df = self._perform_spark_analytics(spark_df)
            result_df = enriched_df.toPandas()
            self.logger.info(f"✅ Spark processing completed: {len(result_df)} records")
            return result_df
        except Exception as e:
            self.logger.error(f"❌ Spark processing failed: {e}")
            return self._pandas_fallback_processing(pandas_df)

    # -----------------------------
    # Spark Analytics
    # -----------------------------
    def _perform_spark_analytics(self, spark_df):
        """Perform advanced Spark analytics on invoice data."""
        # Industry-level analytics
        industry_analytics = spark_df.groupBy("customer_industry") \
            .agg(
                avg("invoice_amount").alias("industry_avg_amount"),
                count("*").alias("industry_invoice_count"),
                avg("customer_credit_score").alias("industry_avg_credit_score"),
                sum(when(col("risk_level") == "High", 1).otherwise(0)).alias("high_risk_count")
            ) \
            .withColumn("high_risk_percentage", col("high_risk_count") / col("industry_invoice_count") * 100)

        # Customer-level segmentation
        customer_segments = spark_df.groupBy("customer_id") \
            .agg(
                count("*").alias("total_invoices"),
                avg("invoice_amount").alias("avg_invoice_amount"),
                avg("customer_credit_score").alias("avg_credit_score"),
                avg("predicted_delay_days").alias("avg_delay_days"),
                sum("invoice_amount").alias("total_business_volume")
            ) \
            .withColumn("customer_segment",
                        when(col("total_business_volume") > 100000, "VIP")
                        .when(col("total_business_volume") > 50000, "Premium")
                        .when(col("total_business_volume") > 10000, "Standard")
                        .otherwise("Small"))

        # Merge analytics with main dataset
        enriched_df = spark_df \
            .join(industry_analytics, "customer_industry", "left") \
            .join(customer_segments, "customer_id", "left")

        return enriched_df

    # -----------------------------
    # Pandas Fallback
    # -----------------------------
    def _pandas_fallback_processing(self, pandas_df: pd.DataFrame) -> pd.DataFrame:
        """Fallback analytics using pandas when Spark is unavailable."""
        self.logger.info("Using pandas fallback for data processing")

        if 'customer_industry' in pandas_df.columns:
            industry_stats = pandas_df.groupby('customer_industry').agg({
                'invoice_amount': ['mean', 'count'],
                'customer_credit_score': 'mean'
            }).round(2)
            industry_stats.columns = ['industry_avg_amount', 'industry_invoice_count', 'industry_avg_credit_score']
            industry_stats = industry_stats.reset_index()
            pandas_df = pandas_df.merge(industry_stats, on='customer_industry', how='left')

        return pandas_df

    # -----------------------------
    # Real-Time Stream Processing
    # -----------------------------
    def process_real_time_stream(self, data_stream):
        """Placeholder for future real-time stream processing (e.g., Kafka)."""
        self.logger.info("Real-time stream processing placeholder")
        return data_stream

    # -----------------------------
    # Advanced Metrics
    # -----------------------------
    def calculate_advanced_metrics(self, spark_df):
        """Calculate global business metrics using Spark."""
        try:
            metrics = spark_df.agg(
                avg("invoice_amount").alias("global_avg_invoice"),
                sum("invoice_amount").alias("total_portfolio_value"),
                avg("predicted_delay_days").alias("avg_predicted_delay"),
                expr("percentile_approx(predicted_delay_days, 0.95)").alias("p95_delay_days"),
                count(when(col("risk_level") == "High", True)).alias("total_high_risk")
            ).collect()[0]

            return {
                'global_avg_invoice': metrics['global_avg_invoice'],
                'total_portfolio_value': metrics['total_portfolio_value'],
                'avg_predicted_delay': metrics['avg_predicted_delay'],
                'p95_delay_days': metrics['p95_delay_days'],
                'total_high_risk': metrics['total_high_risk']
            }
        except Exception as e:
            self.logger.error(f"Advanced metrics calculation failed: {e}")
            return {}

    # -----------------------------
    # Cleanup
    # -----------------------------
    def shutdown(self):
        """Stop Spark session."""
        if self.spark:
            self.spark.stop()
            self.logger.info("Spark session stopped")


# -----------------------------
# Global Spark Processor Instance
# -----------------------------
spark_processor = SparkDataProcessor()
