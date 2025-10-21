# spark_processor.py - Big Data Spark Integration
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd
import logging

class SparkDataProcessor:
    """
    Big Data processor for handling large invoice datasets using Apache Spark
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.spark = None
        self.initialize_spark()
    
    def initialize_spark(self):
        """Initialize Spark session with optimized configuration"""
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
            self.logger.error(f"❌ Failed to initialize Spark: {str(e)}")
            self.spark = None
    
    def process_large_dataset(self, pandas_df, use_spark_threshold=100000):
        """
        Process datasets using Spark for big data operations
        
        Args:
            pandas_df: Input pandas DataFrame
            use_spark_threshold: Use Spark for datasets larger than this
            
        Returns:
            Processed pandas DataFrame
        """
        if self.spark is None:
            self.logger.warning("Spark not available, using pandas fallback")
            return self._pandas_fallback_processing(pandas_df)
        
        try:
            # Convert to Spark DataFrame
            spark_df = self.spark.createDataFrame(pandas_df)
            
            # Big data analytics operations
            processed_df = self._perform_spark_analytics(spark_df)
            
            # Convert back to pandas
            result_df = processed_df.toPandas()
            self.logger.info(f"✅ Spark processing completed: {len(result_df)} records")
            return result_df
            
        except Exception as e:
            self.logger.error(f"❌ Spark processing failed: {str(e)}")
            return self._pandas_fallback_processing(pandas_df)
    
    def _perform_spark_analytics(self, spark_df):
        """Perform advanced Spark analytics on invoice data"""
        
        # Industry risk analysis
        industry_analytics = spark_df \
            .groupBy("customer_industry") \
            .agg(
                avg("invoice_amount").alias("industry_avg_amount"),
                count("*").alias("industry_invoice_count"),
                avg("customer_credit_score").alias("industry_avg_credit_score"),
                sum(when(col("risk_level") == "High", 1).otherwise(0)).alias("high_risk_count")
            ) \
            .withColumn("high_risk_percentage", 
                      col("high_risk_count") / col("industry_invoice_count") * 100)
        
        # Customer segmentation
        customer_segments = spark_df \
            .groupBy("customer_id") \
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
        
        # Join analytics back to main dataframe
        enriched_df = spark_df \
            .join(industry_analytics, "customer_industry", "left") \
            .join(customer_segments, "customer_id", "left")
        
        return enriched_df
    
    def _pandas_fallback_processing(self, pandas_df):
        """Fallback processing when Spark is unavailable"""
        self.logger.info("Using pandas fallback for data processing")
        
        # Basic analytics using pandas
        if 'customer_industry' in pandas_df.columns:
            industry_stats = pandas_df.groupby('customer_industry').agg({
                'invoice_amount': ['mean', 'count'],
                'customer_credit_score': 'mean'
            }).round(2)
            
            # Flatten column names
            industry_stats.columns = ['industry_avg_amount', 'industry_invoice_count', 'industry_avg_credit_score']
            industry_stats = industry_stats.reset_index()
            
            # Merge back
            pandas_df = pandas_df.merge(industry_stats, on='customer_industry', how='left')
        
        return pandas_df
    
    def process_real_time_stream(self, data_stream):
        """Process real-time data streams (placeholder for future implementation)"""
        # This can be extended for Kafka/real-time processing
        self.logger.info("Real-time stream processing placeholder")
        return data_stream
    
    def calculate_advanced_metrics(self, spark_df):
        """Calculate advanced business metrics using Spark"""
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
            self.logger.error(f"Advanced metrics calculation failed: {str(e)}")
            return {}
    
    def shutdown(self):
        """Cleanup Spark session"""
        if self.spark:
            self.spark.stop()
            self.logger.info("Spark session stopped")

# Singleton instance for app-wide usage
spark_processor = SparkDataProcessor()