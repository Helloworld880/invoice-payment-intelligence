import streamlit as st
import plotly.express as px
import pandas as pd

def business_insights(self):
    """Display business insights, financial impact, risk distribution, and recommendations."""
    st.header("💡 Business Insights")

    try:
        # Check if analyzed data is available in session
        analyzed_data = st.session_state.get('analyzed_data', None)

        if analyzed_data is not None:
            st.success(f"📊 Using analyzed data from Batch Analysis - {len(analyzed_data)} invoices")

            # Generate insights
            from business_insights import BusinessInsights
            insights = BusinessInsights()
            summary = insights.generate_executive_summary(analyzed_data)
            financial = summary['financial_impact']

            st.success(f"📈 Analysis completed on {summary['analysis_date']}")

            # -----------------------------
            # Financial Impact Section
            # -----------------------------
            st.subheader("💰 Financial Impact Analysis")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("High Risk Amount", f"${financial['high_risk_amount']:,.0f}")
            col2.metric("Opportunity Cost", f"${financial['opportunity_cost']:,.0f}")
            col3.metric("Estimated Savings", f"${financial['estimated_savings']:,.0f}")
            col4.metric("Cash Flow Impact", f"${financial['cash_flow_impact']:,.0f}")

            # -----------------------------
            # Risk Distribution
            # -----------------------------
            st.subheader("🎯 Risk Distribution")
            col1, col2 = st.columns(2)

            # Risk pie chart
            with col1:
                risk_data = pd.DataFrame({
                    'Risk Level': ['High Risk', 'Medium Risk', 'Low Risk'],
                    'Count': [
                        financial['high_risk_count'],
                        financial['medium_risk_count'],
                        financial['total_invoices'] - financial['high_risk_count'] - financial['medium_risk_count']
                    ]
                })
                fig = px.pie(
                    risk_data,
                    values='Count',
                    names='Risk Level',
                    title="Invoice Risk Distribution",
                    color_discrete_sequence=['#F44336', '#FF9800', '#4CAF50']
                )
                st.plotly_chart(fig, use_container_width=True)

            # Industry-level high risk %
            with col2:
                if summary.get('industry_analysis'):
                    industry_df = pd.DataFrame(summary['industry_analysis'])
                    fig = px.bar(
                        industry_df,
                        x='industry',
                        y='high_risk_percentage',
                        title="High Risk % by Industry",
                        color='high_risk_percentage',
                        color_continuous_scale='RdYlGn_r'
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # -----------------------------
            # Strategic Recommendations
            # -----------------------------
            st.subheader("🚀 Strategic Recommendations")
            recommendations = summary.get('recommendations', [])
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    with st.expander(f"Recommendation {i}: {rec['recommendation']}"):
                        col1, col2, col3 = st.columns(3)
                        col1.write(f"**Priority:** {rec['priority']}")
                        col2.write(f"**Category:** {rec['category']}")
                        col3.write(f"**Timeline:** {rec['timeline']}")
            else:
                st.info("No specific recommendations generated.")

            # -----------------------------
            # Performance Metrics
            # -----------------------------
            st.subheader("📈 Performance Metrics")
            metrics = summary['performance_metrics']
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Delay Days", f"{metrics['average_delay_days']:.1f}")
            col2.metric("On-Time Rate", f"{metrics['on_time_percentage']:.1f}%")
            col3.metric("High Risk %", f"{financial['high_risk_percentage']:.1f}%")
            col4.metric("Total Analyzed", f"{financial['total_invoices']}")

        else:
            st.info("📊 No analyzed data found. Please upload and analyze data in Batch Analysis first.")

            # Show demo insights with sample data
            if st.checkbox("Show sample insights with demo data"):
                sample_data = self.load_sample_data()
                from business_insights import BusinessInsights
                insights = BusinessInsights()
                summary = insights.generate_executive_summary(sample_data)
                financial = summary['financial_impact']
                st.success("📊 Showing sample insights with demo data")

                # Sample Financial Impact
                st.subheader("💰 Sample Financial Impact Analysis")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("High Risk Amount", f"${financial['high_risk_amount']:,.0f}")
                col2.metric("Opportunity Cost", f"${financial['opportunity_cost']:,.0f}")
                col3.metric("Estimated Savings", f"${financial['estimated_savings']:,.0f}")
                col4.metric("Cash Flow Impact", f"${financial['cash_flow_impact']:,.0f}")

    except Exception as e:
        st.error(f"❌ Error generating insights: {str(e)}")
        st.info("💡 Please upload and analyze data in Batch Analysis first.")
