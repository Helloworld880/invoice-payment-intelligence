import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.utils.config import Config
from src.utils.helpers import calculate_financial_impact, format_currency

class InsightsComponents:
    """Business insights components for the Streamlit app"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def render_insights_header(self):
        """Render business insights header"""
        st.header("💡 Business Insights")
        st.markdown("Deep analytics and strategic recommendations for your invoice portfolio")
    
    def render_comprehensive_insights(self, data):
        """Render comprehensive business insights"""
        # Financial impact
        financial_impact = calculate_financial_impact(data)
        
        st.subheader("💰 Financial Impact Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("High Risk Amount", format_currency(financial_impact['high_risk_amount']))
        with col2:
            st.metric("Opportunity Cost", format_currency(financial_impact['opportunity_cost']))
        with col3:
            st.metric("Estimated Savings", format_currency(financial_impact['estimated_savings']))
        with col4:
            st.metric("Avg High Risk Delay", f"{financial_impact['avg_high_risk_delay']:.1f} days")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_risk_by_industry(data)
        
        with col2:
            self._render_delay_distribution(data)
    
    def _render_risk_by_industry(self, data):
        """Render risk analysis by industry"""
        if 'customer_industry' in data.columns and 'risk_level' in data.columns:
            industry_risk = data.groupby('customer_industry').agg({
                'risk_level': lambda x: (x == 'High').mean() * 100,
                'invoice_amount': 'sum',
                'predicted_delay_days': 'mean'
            }).reset_index()
            
            industry_risk.columns = ['Industry', 'High Risk %', 'Total Amount', 'Avg Delay Days']
            
            fig = px.bar(industry_risk, x='Industry', y='High Risk %',
                        color='Total Amount', 
                        title="High Risk % by Industry",
                        color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_delay_distribution(self, data):
        """Render delay distribution chart"""
        if 'predicted_delay_days' in data.columns:
            fig = px.histogram(data, x='predicted_delay_days', 
                             title="Distribution of Predicted Delay Days",
                             nbins=20, color_discrete_sequence=['#ff6b6b'])
            fig.update_layout(xaxis_title="Delay Days", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
    
    def render_strategic_recommendations(self, data):
        """Render strategic recommendations"""
        st.subheader("🚀 Strategic Recommendations")
        
        high_risk_count = (data['risk_level'] == 'High').sum()
        high_risk_percentage = (high_risk_count / len(data)) * 100
        high_risk_amount = data[data['risk_level'] == 'High']['invoice_amount'].sum()
        
        recommendations = []
        
        # High risk recommendation
        if high_risk_count > 0:
            recommendations.append({
                'priority': 'High',
                'title': 'Immediate High-Risk Intervention',
                'description': f'Focus on {high_risk_count} high-risk invoices totaling {format_currency(high_risk_amount)}',
                'actions': [
                    'Contact customers for payment confirmation',
                    'Offer early payment discounts',
                    'Escalate to collections if necessary'
                ]
            })
        
        # Industry-specific recommendations
        if 'customer_industry' in data.columns:
            industry_analysis = data.groupby('customer_industry')['risk_level'].apply(
                lambda x: (x == 'High').mean() * 100
            ).sort_values(ascending=False)
            
            for industry, risk_pct in industry_analysis.head(2).items():
                if risk_pct > 25:
                    recommendations.append({
                        'priority': 'Medium',
                        'title': f'Review {industry} Payment Terms',
                        'description': f'{risk_pct:.1f}% of {industry} invoices are high risk',
                        'actions': [
                            'Implement stricter payment terms',
                            'Require advance payments',
                            'Conduct credit reviews'
                        ]
                    })
        
        # General recommendations
        recommendations.extend([
            {
                'priority': 'Medium',
                'title': 'Automate Payment Reminders',
                'description': 'Implement automated follow-up system for high-risk invoices',
                'actions': [
                    'Set up email/SMS reminders',
                    'Create escalation workflows',
                    'Monitor response rates'
                ]
            },
            {
                'priority': 'Low',
                'title': 'Quarterly Portfolio Review',
                'description': 'Regular analysis of customer payment patterns',
                'actions': [
                    'Update credit scores quarterly',
                    'Review industry trends',
                    'Adjust risk thresholds'
                ]
            }
        ])
        
        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"Recommendation {i}: {rec['title']} ({rec['priority']} Priority)"):
                st.write(f"**Description:** {rec['description']}")
                st.write("**Recommended Actions:**")
                for action in rec['actions']:
                    st.write(f"• {action}")
    
    def render_export_section(self, data):
        """Render data export section"""
        st.subheader("📤 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Summary report
            summary_data = {
                'total_invoices': [len(data)],
                'high_risk_count': [(data['risk_level'] == 'High').sum()],
                'total_amount': [data['invoice_amount'].sum()],
                'high_risk_amount': [data[data['risk_level'] == 'High']['invoice_amount'].sum()],
                'analysis_timestamp': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
            }
            summary_df = pd.DataFrame(summary_data)
            
            st.download_button(
                "📊 Download Summary Report",
                data=summary_df.to_csv(index=False),
                file_name="analysis_summary.csv",
                mime="text/csv"
            )
        
        with col2:
            # Full results
            st.download_button(
                "📁 Download Full Results",
                data=data.to_csv(index=False),
                file_name="detailed_analysis.csv",
                mime="text/csv"
            )
    
    def render_sample_insights(self):
        """Render sample insights when no data is available"""
        st.info("📊 No analyzed data found. Please upload and analyze data in Batch Analysis first.")
        st.info("💡 Showing sample insights with demo data...")
        
        # Create sample data for demonstration
        sample_data = self._create_sample_data()
        self.render_comprehensive_insights(sample_data)
    
    def _create_sample_data(self):
        """Create sample data for demonstration"""
        np.random.seed(42)
        industries = ['Technology', 'Manufacturing', 'Retail', 'Healthcare', 'Construction']
        
        return pd.DataFrame({
            'customer_industry': np.random.choice(industries, 50),
            'invoice_amount': np.random.lognormal(9, 1, 50) * 100,
            'risk_level': np.random.choice(['Low', 'Medium', 'High'], 50, p=[0.6, 0.3, 0.1]),
            'predicted_delay_days': np.random.exponential(10, 50)
        })