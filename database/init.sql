-- database/init.sql
CREATE TABLE IF NOT EXISTS invoice_predictions (
    id SERIAL PRIMARY KEY,
    invoice_id VARCHAR(100),
    customer_industry VARCHAR(50),
    invoice_amount DECIMAL(15,2),
    credit_score INTEGER,
    predicted_delay_days DECIMAL(10,2),
    risk_level VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_type VARCHAR(50),
    accuracy DECIMAL(5,4),
    precision DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_invoice_industry ON invoice_predictions(customer_industry);
CREATE INDEX IF NOT EXISTS idx_invoice_risk ON invoice_predictions(risk_level);
CREATE INDEX IF NOT EXISTS idx_invoice_created ON invoice_predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_credit_score ON invoice_predictions(credit_score);