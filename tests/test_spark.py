# tests/test_spark.py - Spark Processor Tests
import pytest
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spark_processor import SparkDataProcessor

class TestSparkProcessor:
    """Test cases for Spark Data Processor"""
    
    def setup_method(self):
        """Setup before each test"""
        self.spark_processor = SparkDataProcessor()
        self.sample_data = pd.DataFrame({
            'customer_industry': ['Technology', 'Manufacturing', 'Retail', 'Technology'],
            'customer_credit_score': [720, 650, 680, 700],
            'invoice_amount': [15000.0, 25000.0, 8000.0, 12000.0],
            'risk_level': ['Low', 'Medium', 'Low', 'High'],
            'customer_id': ['CUST001', 'CUST002', 'CUST003', 'CUST001']
        })
    
    def test_spark_initialization(self):
        """Test Spark session initialization"""
        assert self.spark_processor.spark is not None
        assert self.spark_processor.spark.sparkContext.appName == "InvoicePaymentAnalytics"
    
    def test_process_large_dataset(self):
        """Test dataset processing with Spark"""
        result = self.spark_processor.process_large_dataset(self.sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(self.sample_data)
        assert 'industry_avg_amount' in result.columns
    
    def test_pandas_fallback(self):
        """Test pandas fallback processing"""
        # Simulate Spark failure
        self.spark_processor.spark = None
        result = self.spark_processor.process_large_dataset(self.sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(self.sample_data)
    
    def test_advanced_metrics(self):
        """Test advanced metrics calculation"""
        if self.spark_processor.spark:
            spark_df = self.spark_processor.spark.createDataFrame(self.sample_data)
            metrics = self.spark_processor.calculate_advanced_metrics(spark_df)
            
            assert isinstance(metrics, dict)
            assert 'global_avg_invoice' in metrics
            assert 'total_high_risk' in metrics
    
    def teardown_method(self):
        """Cleanup after each test"""
        if hasattr(self, 'spark_processor') and self.spark_processor.spark:
            self.spark_processor.shutdown()