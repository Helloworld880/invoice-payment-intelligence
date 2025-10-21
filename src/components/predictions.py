import streamlit as st
import pandas as pd
import numpy as np
from src.utils.config import Config
from src.utils.helpers import validate_file_upload, create_demo_predictions, format_currency
from src.utils.validators import DataValidator, InputValidator

class PredictionComponents:
    """Prediction components for the Streamlit app"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_validator = DataValidator(config)
        self.input_validator = InputValidator(config)
    
    def render_single_prediction_header(self):
        """Render single prediction header"""
        st.header("🔮 Single Invoice Prediction")
        st.markdown("Predict payment delay risk for individual invoices")
    
    def render_prediction_form(self, models):
        """Render prediction form and return results"""
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                invoice_amount = st.number_input("Invoice Amount ($)", 
                                               min_value=0.0, 
                                               max_value=1000000.0, 
                                               value=10000.0, 
                                               step=1000.0)
                customer_credit_score = st.slider("Customer Credit Score", 
                                                 min_value=300, 
                                                 max_value=850, 
                                                 value=650)
                due_days = st.selectbox("Payment Terms (Days)", [15, 30, 45, 60, 90])
            
            with col2:
                customer_industry = st.selectbox(
                    "Customer Industry",
                    ['Technology', 'Manufacturing', 'Retail', 'Healthcare', 'Construction', 'Professional Services']
                )
                avg_payment_delay_history = st.number_input("Historical Avg Delay (Days)", 
                                                           min_value=0.0, 
                                                           max_value=365.0, 
                                                           value=10.0, 
                                                           step=1.0)
                payment_consistency = st.slider("Payment Consistency (0-1)", 
                                               min_value=0.0, 
                                               max_value=1.0, 
                                               value=0.8, 
                                               step=0.05)
            
            submitted = st.form_submit_button("🎯 Predict Payment Behavior", type="primary")
            
            if submitted:
                return self._process_prediction({
                    'invoice_amount': invoice_amount,
                    'customer_credit_score': customer_credit_score,
                    'due_days': due_days,
                    'customer_industry': customer_industry,
                    'avg_payment_delay_history': avg_payment_delay_history,
                    'payment_consistency': payment_consistency
                }, models)
        
        return None
    
    def _process_prediction(self, input_data, models):
        """Process single prediction"""
        with st.spinner("🔍 Analyzing invoice patterns..."):
            try:
                # Validate inputs
                is_valid, errors = self.input_validator.validate_single_input(input_data)
                if not is_valid:
                    for error in errors:
                        st.error(error)
                    return None
                
                # Create input DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Make prediction
                if models['classifier'] is not None and models['regressor'] is not None:
                    # Real model prediction
                    processed_input = self._preprocess_input(input_df)
                    delay_prob = models['classifier'].predict_proba(processed_input)[:, 1][0]
                    predicted_delay_days = max(0, models['regressor'].predict(processed_input)[0])
                else:
                    # Demo prediction
                    predictions = create_demo_predictions(input_df)
                    delay_prob = predictions['delay_probs'][0]
                    predicted_delay_days = predictions['predicted_delays'][0]
                
                # Determine risk level
                risk_thresholds = self.config.risk_thresholds
                if delay_prob > risk_thresholds['medium']:
                    risk_level = "High"
                    action = "🚨 Immediate follow-up required"
                elif delay_prob > risk_thresholds['low']:
                    risk_level = "Medium" 
                    action = "⚠️ Monitor closely"
                else:
                    risk_level = "Low"
                    action = "✅ Normal process"
                
                return {
                    'delay_prob': delay_prob,
                    'predicted_delay': predicted_delay_days,
                    'risk_level': risk_level,
                    'action': action,
                    'input_data': input_data
                }
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                return None
    
    def render_prediction_results(self, result):
        """Render prediction results"""
        if not result:
            return
        
        st.success("✅ Prediction completed!")
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Delay Probability", f"{result['delay_prob']:.1%}")
        with col2:
            st.metric("Predicted Delay", f"{result['predicted_delay']:.1f} days")
        with col3:
            risk_color = {
                'High': '#f44336',
                'Medium': '#ff9800', 
                'Low': '#4caf50'
            }
            st.markdown(
                f'<div style="text-align: center; padding: 10px; background-color: {risk_color[result["risk_level"]]}; '
                f'color: white; border-radius: 5px; font-weight: bold;">{result["risk_level"]} Risk</div>',
                unsafe_allow_html=True
            )
        
        # Risk box
        risk_class = f"{result['risk_level'].lower()}-risk"
        st.markdown(f'''
        <div class="prediction-box {risk_class}">
            <h4>🎯 Recommended Action</h4>
            <p>{result['action']}</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Risk factors
        self._render_risk_factors(result['input_data'])
    
    def _render_risk_factors(self, input_data):
        """Render risk factors analysis"""
        st.subheader("🔍 Risk Factors Analysis")
        factors = []
        
        if input_data['customer_credit_score'] < 600:
            factors.append(f"• Low credit score ({input_data['customer_credit_score']})")
        
        if input_data['avg_payment_delay_history'] > 15:
            factors.append(f"• Poor payment history ({input_data['avg_payment_delay_history']} days average delay)")
        
        if input_data['invoice_amount'] > 50000:
            factors.append("• Large invoice amount")
        
        if input_data['customer_industry'] in ['Construction', 'Healthcare']:
            factors.append(f"• {input_data['customer_industry']} industry typically has longer payment cycles")
        
        if factors:
            for factor in factors:
                st.write(factor)
        else:
            st.info("• No significant risk factors identified")
    
    def render_batch_analysis_header(self):
        """Render batch analysis header"""
        st.header("📊 Batch Invoice Analysis")
        st.markdown("Upload a CSV file to analyze multiple invoices at once")
    
    def render_file_uploader(self):
        """Render file uploader"""
        return st.file_uploader(
            "Upload CSV file with invoice data", 
            type=["csv"],
            help="Your CSV should contain: customer_industry, customer_credit_score, invoice_amount"
        )
    
    def process_uploaded_file(self, uploaded_file, models, validator):
        """Process uploaded CSV file"""
        try:
            # Read the file
            batch_data = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! {len(batch_data)} records found.")
            
            # Show preview
            st.subheader("📋 Data Preview")
            st.dataframe(batch_data.head())
            
            # Validate data
            validation_result = validator.validate_user_data(batch_data)
            
            if not validation_result['is_valid']:
                for error in validation_result['errors']:
                    st.error(error)
                return None
            
            if validation_result['warnings']:
                for warning in validation_result['warnings']:
                    st.warning(warning)
            
            # Process data
            cleaned_data = validation_result['cleaned_data']
            
            # Make predictions
            with st.spinner("Processing batch data..."):
                progress_bar = st.progress(0)
                
                if models['classifier'] is not None and models['regressor'] is not None:
                    # Real model predictions
                    processed_data = self._preprocess_input(cleaned_data)
                    delay_probs = models['classifier'].predict_proba(processed_data)[:, 1]
                    predicted_delays = models['regressor'].predict(processed_data)
                else:
                    # Demo predictions
                    predictions = create_demo_predictions(cleaned_data)
                    delay_probs = predictions['delay_probs']
                    predicted_delays = predictions['predicted_delays']
                
                progress_bar.progress(100)
                
                # Add predictions to results
                results = cleaned_data.copy()
                results['delay_probability'] = delay_probs
                results['predicted_delay_days'] = [max(0, x) for x in predicted_delays]
                
                # Assign risk levels
                risk_thresholds = self.config.risk_thresholds
                results['risk_level'] = pd.cut(
                    results['delay_probability'], 
                    bins=[0, risk_thresholds['low'], risk_thresholds['medium'], 1], 
                    labels=['Low', 'Medium', 'High']
                )
                
                results['recommended_action'] = results['risk_level'].map({
                    'Low': 'Normal process',
                    'Medium': 'Monitor closely',
                    'High': 'Immediate follow-up'
                })
                
                return results
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            return None
    
    def render_batch_results(self, results):
        """Render batch analysis results"""
        st.success(f"✅ Analysis complete! {len(results)} invoices processed.")
        
        # Summary metrics
        high_risk_count = (results['risk_level'] == 'High').sum()
        medium_risk_count = (results['risk_level'] == 'Medium').sum()
        total_amount = results['invoice_amount'].sum()
        high_risk_amount = results[results['risk_level'] == 'High']['invoice_amount'].sum()
        
        st.subheader("📈 Analysis Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Invoices", len(results))
        with col2:
            st.metric("High Risk", high_risk_count)
        with col3:
            st.metric("Medium Risk", medium_risk_count)
        with col4:
            st.metric("High Risk Amount", format_currency(high_risk_amount))
        
        # Show results table
        st.subheader("📋 Detailed Predictions")
        display_cols = ['invoice_amount', 'customer_industry', 'delay_probability', 
                      'predicted_delay_days', 'risk_level', 'recommended_action']
        available_cols = [col for col in display_cols if col in results.columns]
        st.dataframe(results[available_cols].head(20))
    
    def _preprocess_input(self, input_data):
        """Preprocess input data for model prediction"""
        processed = input_data.copy()
        
        # Handle categorical features
        if 'customer_industry' in processed.columns:
            industry_dummies = pd.get_dummies(processed['customer_industry'], prefix='industry')
            possible_industries = ['Technology', 'Manufacturing', 'Retail', 'Healthcare', 'Construction', 'Professional Services']
            
            for industry in possible_industries:
                col_name = f'industry_{industry}'
                if col_name in industry_dummies.columns:
                    processed[col_name] = industry_dummies[col_name]
                else:
                    processed[col_name] = 0
        
        return processed