import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_realistic_invoice_data():
    """Generate realistic invoice payment dataset for business analysis"""
    print("📊 Generating realistic invoice payment dataset...")

    np.random.seed(42)

    # Business parameters
    industries = ['Technology', 'Manufacturing', 'Retail', 'Healthcare', 'Construction', 'Professional Services']
    industry_risk = {
        'Technology': {'avg_delay': 8, 'credit_mean': 720},
        'Manufacturing': {'avg_delay': 15, 'credit_mean': 650},
        'Retail': {'avg_delay': 12, 'credit_mean': 680},
        'Healthcare': {'avg_delay': 18, 'credit_mean': 620},
        'Construction': {'avg_delay': 22, 'credit_mean': 580},
        'Professional Services': {'avg_delay': 10, 'credit_mean': 700}
    }

    customer_segments = {
        'Excellent': {'prob': 0.15, 'credit_range': (750, 850), 'delay_multiplier': 0.3},
        'Good': {'prob': 0.25, 'credit_range': (680, 749), 'delay_multiplier': 0.6},
        'Average': {'prob': 0.35, 'credit_range': (600, 679), 'delay_multiplier': 1.0},
        'Poor': {'prob': 0.20, 'credit_range': (500, 599), 'delay_multiplier': 1.5},
        'Risky': {'prob': 0.05, 'credit_range': (300, 499), 'delay_multiplier': 2.0}
    }

    # Generate 5000 realistic invoices
    invoices = []
    customer_profiles = {}

    for i in range(5000):
        # Select industry and customer segment
        industry = np.random.choice(industries)
        segment = np.random.choice(list(customer_segments.keys()), 
                                 p=[v['prob'] for v in customer_segments.values()])

        # Create customer profile if new customer
        customer_id = f"CUST_{np.random.randint(1000, 9999)}"
        if customer_id not in customer_profiles:
            credit_range = customer_segments[segment]['credit_range']
            customer_profiles[customer_id] = {
                'industry': industry,
                'segment': segment,
                'base_credit': np.random.randint(credit_range[0], credit_range[1]),
                'payment_consistency': np.random.beta(8, 2)  # Most customers are consistent
            }

        profile = customer_profiles[customer_id]

        # Generate invoice details
        invoice_amount = np.random.lognormal(9, 1.2)  # Realistic business amounts
        issue_date = datetime(2023, 1, 1) + timedelta(days=int(np.random.randint(0, 365)))
        due_days = int(np.random.choice([15, 30, 45, 60], p=[0.1, 0.6, 0.2, 0.1]))  # FIXED: convert to int
        due_date = issue_date + timedelta(days=due_days)

        # Calculate realistic payment delay
        base_delay = industry_risk[industry]['avg_delay']
        segment_multiplier = customer_segments[segment]['delay_multiplier']
        consistency_effect = 1.5 - profile['payment_consistency']  # Lower consistency = more delay

        # Amount effect (larger invoices paid slower)
        amount_effect = 1 + (min(1, invoice_amount / 100000) * 0.4)

        # Seasonal effect (year-end delays)
        seasonal_effect = 1.3 if issue_date.month in [11, 12] else 1.0

        # Final delay calculation
        actual_delay = base_delay * segment_multiplier * consistency_effect * amount_effect * seasonal_effect
        actual_delay += np.random.normal(0, 3)  # Add randomness
        actual_delay = max(-7, actual_delay)  # Allow early payments

        payment_date = due_date + timedelta(days=int(actual_delay))

        invoices.append({
            'invoice_id': f'INV_{100000 + i}',
            'customer_id': customer_id,
            'customer_industry': industry,
            'customer_segment': segment,
            'customer_credit_score': max(300, min(850, profile['base_credit'] + np.random.randint(-20, 20))),
            'invoice_amount': round(invoice_amount, 2),
            'issue_date': issue_date.strftime('%Y-%m-%d'),
            'due_date': due_date.strftime('%Y-%m-%d'),
            'payment_date': payment_date.strftime('%Y-%m-%d'),
            'payment_delay_days': round(actual_delay, 1),
            'is_delayed': 1 if actual_delay > 0 else 0,
            'due_days': due_days,
            'avg_payment_delay_history': round(base_delay * segment_multiplier, 1),
            'payment_consistency': round(profile['payment_consistency'], 3),
            'quarter': (issue_date.month - 1) // 3 + 1,
            'month': issue_date.month,
            'is_weekend_issue': 1 if issue_date.weekday() >= 5 else 0
        })

    # Create DataFrame
    df = pd.DataFrame(invoices)

    # Save to CSV
    os.makedirs('data/raw', exist_ok=True)
    output_path = 'data/raw/invoice_data.csv'
    df.to_csv(output_path, index=False)

    print("✅ Dataset generated successfully!")
    print(f"📁 File saved at: {output_path}")
    print(f"📊 Total invoices: {len(df):,}")
    print(f"⏰ Delayed invoices: {df['is_delayed'].sum():,} ({df['is_delayed'].mean():.1%})")
    print(f"📅 Average delay (for delayed only): {df[df['is_delayed'] == 1]['payment_delay_days'].mean():.1f} days")
    print(f"🏢 Industries: {', '.join(industries)}")
    print(f"👥 Customer segments: {', '.join(customer_segments.keys())}")
    
    print("\n🔍 Sample preview:")
    print(df.head())

    return df


if __name__ == "__main__":
    generate_realistic_invoice_data()
