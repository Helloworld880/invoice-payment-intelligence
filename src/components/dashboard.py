import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.utils.config import Config
from src.utils.helpers import format_currency

class DashboardComponents:
    """Dashboard components for the Streamlit app"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown("""
        <style>
        .dashboard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        </style>
        <div class="dashboard-header">
            <h1>📊 Executive Dashboard</h1>
            <p>Real-time insights into invoice payment performance and risk analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_key_metrics(self, data=None):
        """Render key performance metrics"""
        st.subheader("📈 Key Metrics")
        
        if data is not None:
            total_invoices = len(data)
            high_risk_count = (data['risk_level'] == 'High').sum() if 'risk_level' in data.columns else 0
            total_amount = data['invoice_amount'].sum() if 'invoice_amount' in data.columns else 0
            avg_delay = data['predicted_delay_days'].mean() if 'predicted_delay_days' in data.columns else 0
        else:
            # Demo data
            total_invoices = 156
            high_risk_count = 23
            total_amount = 2450000
            avg_delay = 8.5
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Invoices", f"{total_invoices:,}")
        
        with col2:
            st.metric("High Risk", f"{high_risk_count}", "-5%")
        
        with col3:
            st.metric("Total Amount", format_currency(total_amount))
        
        with col4:
            st.metric("Avg Delay", f"{avg_delay:.1f} days", "-1.2 days")
    
    def render_risk_distribution(self, data=None):
        """Render risk distribution chart"""
        st.subheader("🎯 Risk Distribution")
        
        if data is not None and 'risk_level' in data.columns:
            risk_data = data['risk_level'].value_counts().reset_index()
            risk_data.columns = ['Risk Level', 'Count']
        else:
            # Demo data
            risk_data = pd.DataFrame({
                'Risk Level': ['Low', 'Medium', 'High'],
                'Count': [65, 23, 12]
            })
        
        fig = px.pie(risk_data, values='Count', names='Risk Level',
                    color_discrete_sequence=['#4CAF50', '#FF9800', '#F44336'],
                    hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    def render_industry_analysis(self, data=None):
        """Render industry analysis chart"""
        st.subheader("🏢 Industry Analysis")
        
        if data is not None and 'customer_industry' in data.columns:
            industry_data = data.groupby('customer_industry').agg({
                'invoice_amount': 'sum',
                'predicted_delay_days': 'mean' if 'predicted_delay_days' in data.columns else None
            }).reset_index()
        else:
            # Demo data
            industry_data = pd.DataFrame({
                'customer_industry': ['Technology', 'Manufacturing', 'Retail', 'Healthcare', 'Construction'],
                'invoice_amount': [500000, 750000, 300000, 600000, 450000],
                'predicted_delay_days': [8.2, 15.5, 12.1, 18.3, 22.7]
            })
        
        fig = px.bar(industry_data, x='customer_industry', y='invoice_amount',
                    color='predicted_delay_days', 
                    title="Total Invoice Amount by Industry",
                    color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    def render_recent_activity(self):
        """Render recent activity section"""
        st.subheader("📋 Recent Activity")
        
        # Sample recent activity data
        activity_data = pd.DataFrame({
            'Timestamp': ['2024-01-15 10:30', '2024-01-15 09:15', '2024-01-14 16:45'],
            'Activity': ['Batch analysis completed - 156 invoices', 
                        'High-risk alert - Construction industry', 
                        'Model retrained - Accuracy improved to 87.2%'],
            'Status': ['Completed', 'Alert', 'Success']
        })
        
        for _, row in activity_data.iterrows():
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{row['Activity']}**")
                    st.caption(row['Timestamp'])
                with col2:
                    status_color = {
                        'Completed': 'green',
                        'Alert': 'red', 
                        'Success': 'blue'
                    }
                    st.markdown(f"<span style='color: {status_color[row['Status']]}; font-weight: bold;'>{row['Status']}</span>", 
                               unsafe_allow_html=True)
                st.divider()