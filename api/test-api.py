import requests

url = "http://localhost:5000/predict"

data = {
    "customer_id": 1001,
    "transaction_amount": 5000.00,
    "age": 25,
    "gender": "Male",
    "income_bracket": "Low",
    "account_age_months": 2,
    "credit_score": 580,
    "account_balance": 100.00,
    "num_credit_cards": 5,
    "employment_status": "Student",
    "home_ownership": "Rent",
    "is_fraud_history": True,
    "risk_category": "High",
    "merchant_category": "Online",
    "is_high_risk": True,
    "avg_transaction_amount": 50.00,
    "country": "US",
    "fraud_reports": 20,
    "established_year": 2020,
    "is_online": True,
    "is_international": True,
    "transaction_type": "Credit",
    "device_type": "Mobile",
    "ip_location": "CN",
    "previous_transactions_24h": 8,
    "amount_to_avg_ratio": 100.0,
    "days_since_last_txn": 0
}

response = requests.post(url, json=data)
print(response.json())