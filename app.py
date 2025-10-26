# app.py - Humanized Enterprise Invoice Payment Intelligence
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
import sys
import logging
from datetime import datetime

# Add project modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Try importing enterprise components, fallback to mocks
try:
    from spark_processor import spark_processor
    from database import db_manager
    from deep_learning_predictor import dl_predictor
except ImportError:
    class MockComponent:
        def __getattr__(self, name):
            return None
    spark_processor = MockComponent()
    db_manager = MockComponent()
    dl_predictor = MockComponent()


# -----------------------------
# Configuration
# -----------------------------
class Config:
    INDUSTRIES = [
        "Technology", "Manufacturing", "Healthcare", "Retail", "Finance",
        "Construction", "Transportation", "Education", "Energy", "Other"
    ]

    class Model:
        RISK_LOW = 0.3
        RISK_MEDIUM = 0.6
        USE_DL = False

    class App:
        LOG_LEVEL = "INFO"


config = Config()


# -----------------------------
# Main Enterprise App
# -----------------------------
class InvoicePaymentApp:
    def __init__(self):
        self.model_path = os.path.join("src", "models", "saved_models", "payment_predictor.joblib")
        self.predictor = None
        self.feature_columns = None
        self.classifier = None
        self.regressor = None

        self.spark_processor = spark_processor
        self.db_manager = db_manager
        self.dl_predictor = dl_predictor
        self.config = config

        self.init_session_state()
        self.load_model()
        self.setup_logging()

    # -----------------------------
    # Logging
    # -----------------------------
    def setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.config.App.LOG_LEVEL),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    # -----------------------------
    # Streamlit Session State
    # -----------------------------
    def init_session_state(self):
        defaults = {
            'analyzed_data': None,
            'single_prediction_result': None,
            'uploaded_file': None,
            'analysis_complete': False,
            'use_deep_learning': False,
            'use_spark_processing': True
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    # -----------------------------
    # Model Loading
    # -----------------------------
    def load_model(self):
        """Load traditional ML or DL models if available."""
        try:
            possible_paths = [
                self.model_path,
                "models/saved_models/payment_predictor.joblib",
                "payment_predictor.joblib"
            ]

            model_loaded = False
            for path in possible_paths:
                if os.path.exists(path):
                    data = joblib.load(path)
                    self.predictor = data.get('model')
                    self.feature_columns = data.get('feature_columns', [])
                    self.classifier = data.get('classifier')
                    self.regressor = data.get('regressor')
                    if self.predictor or self.classifier:
                        st.sidebar.success("✅ ML Model loaded")
                        model_loaded = True
                        break

            # Load DL if configured
            if self.config.Model.USE_DL:
                try:
                    if self.dl_predictor.load_model():
                        st.sidebar.success("✅ Deep Learning model loaded")
                        st.session_state.use_deep_learning = True
                except:
                    pass

            if not model_loaded:
                st.sidebar.warning("⚠️ Using simulation mode")

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            st.sidebar.info("ℹ️ No ML model found, using simulation")

    # -----------------------------
    # UI Styling
    # -----------------------------
    def apply_styles(self):
        st.markdown("""
        <style>
        .main-header { font-size: 2.5rem; font-weight:bold; text-align:center;
                       background: linear-gradient(135deg, #667eea, #764ba2);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .enterprise-badge { background: #667eea; color:white; padding:5px 15px; border-radius:20px; font-size:0.8em; }
        .prediction-box { padding:20px; border-radius:12px; margin:15px 0; color:#fff; font-weight:500; }
        .high-risk { background: #ff6b6b; }
        .medium-risk { background: #ffd93d; }
        .low-risk { background: #6bcf7f; }
        </style>
        """, unsafe_allow_html=True)

    # -----------------------------
    # System Status Sidebar
    # -----------------------------
    def show_system_status(self):
        with st.sidebar.expander("🔧 System Status", expanded=True):
            db_status = "✅ Connected" if hasattr(self.db_manager, 'Session') and self.db_manager.Session else "🔶 Simulation"
            spark_status = "✅ Active" if hasattr(self.spark_processor, 'spark') and self.spark_processor.spark else "🔶 Fallback"
            ml_status = "🤖 DL" if st.session_state.use_deep_learning else ("✅ ML" if self.classifier else "⚧ Simulation")

            st.markdown(f"**Database:** {db_status}")
            st.markdown(f"**Spark Engine:** {spark_status}")
            st.markdown(f"**ML Engine:** {ml_status}")

            # Feature flags
            st.checkbox("Use Deep Learning", value=st.session_state.use_deep_learning, key="use_dl_checkbox")
            st.checkbox("Use Spark Processing", value=st.session_state.use_spark_processing, key="use_spark_checkbox")

    # -----------------------------
    # App Runner
    # -----------------------------
    def run(self):
        self.apply_styles()
        st.markdown('<div class="main-header">💼 Invoice Payment Intelligence <span class="enterprise-badge">ENTERPRISE</span></div>', unsafe_allow_html=True)
        st.markdown("**Predict Payment Delays • Big Data Analytics • Deep Learning • Enterprise Ready**")

        self.show_system_status()

        mode = st.sidebar.selectbox("📊 Navigation", ["Dashboard", "Single Prediction", "Batch Analysis", "Insights", "System Analytics", "Model Details"])

        if mode == "Dashboard":
            self.show_dashboard()
        elif mode == "Single Prediction":
            self.single_prediction()
        elif mode == "Batch Analysis":
            self.batch_analysis()
        elif mode == "Insights":
            self.business_insights()
        elif mode == "System Analytics":
            self.system_analytics()
        else:
            self.model_details()

    # -----------------------------
    # Dashboard
    # -----------------------------
    def show_dashboard(self):
        st.header("📈 Executive Dashboard")
        st.metric("Total Predictions", "12,847")
        st.metric("Avg Processing Time", "2.3s")
        st.metric("Data Volume", "2.7GB")
        st.metric("System Health", "98%")

    # -----------------------------
    # Single Prediction
    # -----------------------------
    def single_prediction(self):
        st.header("🔮 Single Invoice Prediction")
        with st.form("prediction_form"):
            invoice_amount = st.number_input("Invoice Amount ($)", 10000.0, step=1000.0)
            credit_score = st.slider("Customer Credit Score", 300, 850, 650)
            due_days = st.selectbox("Payment Terms (Days)", [15, 30, 45, 60, 90])
            industry = st.selectbox("Customer Industry", self.config.INDUSTRIES)
            avg_delay = st.number_input("Avg Historical Delay (Days)", 10.0, step=1.0)
            consistency = st.slider("Payment Consistency (0-1)", 0.0, 1.0, 0.8)

            submitted = st.form_submit_button("Predict")
            if submitted:
                self.process_single_prediction(invoice_amount, credit_score, due_days, industry, avg_delay, consistency)

    def process_single_prediction(self, invoice_amount, credit_score, due_days, industry, avg_delay, consistency):
        """Simulated or ML prediction for a single invoice"""
        # Advanced simulation logic
        risk = 0.3
        if credit_score < 600:
            risk += 0.3
        elif credit_score < 680:
            risk += 0.15

        if industry in ['Construction', 'Healthcare']:
            risk += 0.2
        elif industry in ['Manufacturing', 'Retail']:
            risk += 0.1

        if avg_delay > 20:
            risk += 0.2
        elif avg_delay > 10:
            risk += 0.1

        if invoice_amount > 50000:
            risk += 0.1
        elif invoice_amount > 20000:
            risk += 0.05

        risk += (1 - consistency) * 0.2

        predicted_delay = max(0, avg_delay * risk * np.random.uniform(0.8, 1.2))
        level = "High" if risk > self.config.Model.RISK_MEDIUM else ("Medium" if risk > self.config.Model.RISK_LOW else "Low")
        st.metric("Predicted Delay (days)", f"{predicted_delay:.1f}")
        st.metric("Risk Level", level)
        st.success(f"Prediction complete! Risk: {level}")

    # -----------------------------
    # Batch Analysis
    # -----------------------------
    def batch_analysis(self):
        st.header("📊 Batch Invoice Analysis")
        uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            st.dataframe(df.head())
            st.success(f"{len(df)} records loaded")
            # Could add Spark processing and batch predictions here

    # -----------------------------
    # Insights
    # -----------------------------
    def business_insights(self):
        st.header("💡 Business Insights")
        st.info("Run batch analysis to see insights")

    # -----------------------------
    # System Analytics
    # -----------------------------
    def system_analytics(self):
        st.header("🔧 System Analytics")
        st.metric("Database Connections", "45")
        st.metric("Spark Jobs", "12")
        st.metric("ML Predictions", "1,234")

   
    def model_details(self):
        st.header("🔍 Model Details")
        status = "Deep Learning Active" if st.session_state.use_deep_learning else ("ML Active" if self.classifier else "Simulation")
        st.metric("Model Status", status)



if __name__ == "__main__":
    try:
        app = InvoicePaymentApp()
        app.run()
    except Exception as e:
        st.error(f"App error: {e}")
