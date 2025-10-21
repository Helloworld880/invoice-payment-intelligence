import streamlit as st
import plotly.express as px
import pandas as pd
def business_insights(self):
    st.header("💡 Business Insights")
    
    try:
        # Check if we have analyzed data from Batch Analysis
        if st.session_state.analyzed_data is not None:
            analyzed_data = st.session_state.analyzed_data
            st.success(f"📊 Using analyzed data from Batch Analysis - {len(analyzed_data)} invoices")
            
            # Initialize insights generator
            from business_insights import BusinessInsights
            insights = BusinessInsights()
            
            # Generate insights from the analyzed data
            summary = insights.generate_executive_summary(analyzed_data)
            financial = summary['financial_impact']
            
            st.success(f"📈 Analysis completed on {summary['analysis_date']}")
            
            # Financial Impact Section
            st.subheader("💰 Financial Impact Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("High Risk Amount", f"${financial['high_risk_amount']:,.0f}")
            with col2:
                st.metric("Opportunity Cost", f"${financial['opportunity_cost']:,.0f}")
            with col3:
                st.metric("Estimated Savings", f"${financial['estimated_savings']:,.0f}")
            with col4:
                st.metric("Cash Flow Impact", f"${financial['cash_flow_impact']:,.0f}")
            
            # Risk Distribution
            st.subheader("🎯 Risk Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                risk_data = pd.DataFrame({
                    'Risk Level': ['High Risk', 'Medium Risk', 'Low Risk'],
                    'Count': [financial['high_risk_count'], financial['medium_risk_count'], 
                             financial['total_invoices'] - financial['high_risk_count'] - financial['medium_risk_count']]
                })
                fig = px.pie(risk_data, values='Count', names='Risk Level',
                            title="Invoice Risk Distribution",
                            color_discrete_sequence=['#F44336', '#FF9800', '#4CAF50'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Industry Analysis
                if summary['industry_analysis']:
                    industry_df = pd.DataFrame(summary['industry_analysis'])
                    fig = px.bar(industry_df, x='industry', y='high_risk_percentage',
                                title="High Risk % by Industry",
                                color='high_risk_percentage',
                                color_continuous_scale='RdYlGn_r')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Strategic Recommendations
            st.subheader("🚀 Strategic Recommendations")
            
            if summary['recommendations']:
                for i, rec in enumerate(summary['recommendations'], 1):
                    with st.expander(f"Recommendation {i}: {rec['recommendation']}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Priority:** {rec['priority']}")
                        with col2:
                            st.write(f"**Category:** {rec['category']}")
                        with col3:
                            st.write(f"**Timeline:** {rec['timeline']}")
            else:
                st.info("No specific recommendations generated.")
                
            # Performance Metrics
            st.subheader("📈 Performance Metrics")
            metrics = summary['performance_metrics']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Delay Days", f"{metrics['average_delay_days']:.1f}")
            with col2:
                st.metric("On-Time Rate", f"{metrics['on_time_percentage']:.1f}%")
            with col3:
                st.metric("High Risk %", f"{financial['high_risk_percentage']:.1f}%")
            with col4:
                st.metric("Total Analyzed", f"{financial['total_invoices']}")
                
        else:
            # Show demo data if no analyzed data exists
            st.info("📊 No analyzed data found. Please upload and analyze data in Batch Analysis first.")
            
            # Optional: Show sample insights with demo data
            if st.checkbox("Show sample insights with demo data"):
                sample_data = self.load_sample_data()
                
                from business_insights import BusinessInsights
                insights = BusinessInsights()
                summary = insights.generate_executive_summary(sample_data)
                financial = summary['financial_impact']
                
                st.success("📊 Showing sample insights with demo data")
                
                # Financial Impact Section
                st.subheader("💰 Sample Financial Impact Analysis")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("High Risk Amount", f"${financial['high_risk_amount']:,.0f}")
                with col2:
                    st.metric("Opportunity Cost", f"${financial['opportunity_cost']:,.0f}")
                with col3:
                    st.metric("Estimated Savings", f"${financial['estimated_savings']:,.0f}")
                with col4:
                    st.metric("Cash Flow Impact", f"${financial['cash_flow_impact']:,.0f}")
            
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        st.info("💡 Please upload and analyze data in Batch Analysis first")