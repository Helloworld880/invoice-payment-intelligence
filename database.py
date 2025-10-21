# database.py - PostgreSQL Database Integration
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import pandas as pd
import logging

Base = declarative_base()

class InvoicePrediction(Base):
    __tablename__ = 'invoice_predictions'
    
    id = Column(Integer, primary_key=True)
    invoice_id = Column(String(100))
    customer_industry = Column(String(50))
    invoice_amount = Column(Float)
    credit_score = Column(Integer)
    predicted_delay_days = Column(Float)
    risk_level = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelMetrics(Base):
    __tablename__ = 'model_metrics'
    
    id = Column(Integer, primary_key=True)
    model_type = Column(String(50))
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    training_date = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.engine = None
        self.Session = None
        self.connect()
    
    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            # Use environment variable or default to SQLite for development
            database_url = os.getenv('DATABASE_URL', 'sqlite:///invoices.db')
            self.engine = create_engine(database_url)
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            self.logger.info("✅ Database connected successfully")
        except Exception as e:
            self.logger.error(f"❌ Database connection failed: {str(e)}")
    
    def save_prediction(self, prediction_data):
        """Save a single prediction to database"""
        if not self.Session:
            self.logger.error("Database not connected")
            return None
        
        session = self.Session()
        try:
            prediction = InvoicePrediction(**prediction_data)
            session.add(prediction)
            session.commit()
            self.logger.info(f"✅ Prediction saved with ID: {prediction.id}")
            return prediction.id
        except Exception as e:
            session.rollback()
            self.logger.error(f"❌ Failed to save prediction: {str(e)}")
            return None
        finally:
            session.close()
    
    def save_batch_predictions(self, predictions_df):
        """Save batch predictions to database"""
        if not self.Session:
            self.logger.error("Database not connected")
            return False
        
        session = self.Session()
        try:
            for _, row in predictions_df.iterrows():
                prediction_data = {
                    'invoice_id': row.get('invoice_id', 'N/A'),
                    'customer_industry': row.get('customer_industry', 'Unknown'),
                    'invoice_amount': row.get('invoice_amount', 0),
                    'credit_score': row.get('customer_credit_score', 650),
                    'predicted_delay_days': row.get('predicted_delay_days', 0),
                    'risk_level': row.get('risk_level', 'Unknown')
                }
                prediction = InvoicePrediction(**prediction_data)
                session.add(prediction)
            
            session.commit()
            self.logger.info(f"✅ Saved {len(predictions_df)} predictions to database")
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"❌ Failed to save batch predictions: {str(e)}")
            return False
        finally:
            session.close()
    
    def get_historical_patterns(self):
        """Get historical patterns using SQL analytics"""
        if not self.Session:
            self.logger.error("Database not connected")
            return pd.DataFrame()
        
        session = self.Session()
        try:
            # Advanced SQL query for business insights
            query = """
            SELECT 
                customer_industry,
                AVG(predicted_delay_days) as avg_delay_days,
                COUNT(*) as total_invoices,
                SUM(CASE WHEN risk_level = 'High' THEN 1 ELSE 0 END) as high_risk_count,
                AVG(invoice_amount) as avg_invoice_amount,
                AVG(credit_score) as avg_credit_score
            FROM invoice_predictions 
            GROUP BY customer_industry
            ORDER BY avg_delay_days DESC
            """
            result = session.execute(query)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            self.logger.info("✅ Historical patterns retrieved successfully")
            return df
        except Exception as e:
            self.logger.error(f"❌ Failed to get historical patterns: {str(e)}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def get_customer_risk_profiles(self):
        """Get customer risk profiles from historical data"""
        if not self.Session:
            return pd.DataFrame()
        
        session = self.Session()
        try:
            query = """
            SELECT 
                customer_industry,
                risk_level,
                COUNT(*) as count,
                AVG(predicted_delay_days) as avg_delay,
                AVG(credit_score) as avg_credit_score
            FROM invoice_predictions 
            GROUP BY customer_industry, risk_level
            ORDER BY customer_industry, risk_level
            """
            result = session.execute(query)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df
        except Exception as e:
            self.logger.error(f"Failed to get customer risk profiles: {str(e)}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def save_model_metrics(self, metrics_data):
        """Save model performance metrics"""
        if not self.Session:
            return False
        
        session = self.Session()
        try:
            metrics = ModelMetrics(**metrics_data)
            session.add(metrics)
            session.commit()
            self.logger.info("✅ Model metrics saved successfully")
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"❌ Failed to save model metrics: {str(e)}")
            return False
        finally:
            session.close()

# Global database instance
db_manager = DatabaseManager()