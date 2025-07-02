import pandas as pd
import numpy as np
import uuid
import random
from datetime import datetime, timedelta
from faker import Faker
from tqdm import tqdm
from pathlib import Path

fake = Faker()
np.random.seed(11)

NUM_CUSTOMERS = 10_000#150_000
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2025, 5, 30)
PRODUCTS = ['checking', 'savings', 'credit_card', 'loan', 'debit']

def generate_customer_profiles(num_customers):
    profiles = []
    for _ in tqdm(range(num_customers), desc="Generating customer profiles"):
        # Ensure join_date is within our simulation window
        join_date = fake.date_between(start_date=START_DATE.date(), end_date=(END_DATE - timedelta(days=30)).date())
        
        profile = {
            "customer_id": str(uuid.uuid4()),
            "age": np.random.randint(18, 85),
            "state": fake.state_abbr(),
            "has_credit_card": np.random.choice([0, 1]),
            "has_loan": np.random.choice([0, 1]),
            "has_checking": 1,
            "has_savings": np.random.choice([0, 1]),
            "join_date": join_date,
            "gender": np.random.choice(['M', 'F', 'Other']),
            "income_bracket": np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
        }
        profiles.append(profile)
    return pd.DataFrame(profiles)

def determine_churn_status(customer_row):
    """Determine if and when a customer churned based on their profile"""
    # Convert join_date to datetime if it's not already
    if isinstance(customer_row['join_date'], str):
        join_date = datetime.strptime(customer_row['join_date'], '%Y-%m-%d')
    else:
        join_date = datetime.combine(customer_row['join_date'], datetime.min.time())
    
    # Base churn probability (20% of customers churn)
    base_churn_prob = 0.2
    
    # Adjust churn probability based on customer characteristics
    churn_prob = base_churn_prob
    if customer_row['age'] < 25:  # Young customers more likely to churn
        churn_prob += 0.1
    if customer_row['has_credit_card'] == 0:  # Less engaged customers
        churn_prob += 0.05
    if customer_row['has_loan'] == 1:  # Loan customers less likely to churn
        churn_prob -= 0.1
    if customer_row['income_bracket'] == 'low':
        churn_prob += 0.05
    
    churn_prob = max(0.05, min(0.4, churn_prob))  # Keep between 5% and 40%
    
    will_churn = np.random.random() < churn_prob
    
    if will_churn:
        # Customer churns between 30 days after joining and 90 days before END_DATE
        min_churn_date = join_date + timedelta(days=30)
        max_churn_date = END_DATE - timedelta(days=90)
        
        if min_churn_date < max_churn_date:
            days_range = (max_churn_date - min_churn_date).days
            churn_date = min_churn_date + timedelta(days=np.random.randint(0, days_range))
            return True, churn_date
    
    return False, None

def generate_transactions(customers, avg_tx_per_customer=100):
    txns = []
    
    for idx, row in tqdm(customers.iterrows(), total=len(customers), desc="Generating transactions"):
        # Determine churn status
        will_churn, churn_date = determine_churn_status(row)
        
        # Set customer's active period
        join_date = datetime.combine(row['join_date'], datetime.min.time()) if isinstance(row['join_date'], type(datetime.now().date())) else row['join_date']
        end_date = churn_date if will_churn else END_DATE
        
        # Skip if join date is after end date
        if join_date >= end_date:
            continue
            
        # Determine which products the customer has
        available_products = ["checking", "debit"]  # always included
        if row["has_credit_card"]:
            available_products.append("credit_card")
        if row["has_loan"]:
            available_products.append("loan")
        if row["has_savings"]:
            available_products.append("savings")

        # Adjust transaction volume based on customer characteristics
        tx_multiplier = 1.0
        if row['income_bracket'] == 'high':
            tx_multiplier = 1.5
        elif row['income_bracket'] == 'low':
            tx_multiplier = 0.7
        
        n_tx = int(np.random.poisson(avg_tx_per_customer * tx_multiplier))
        
        # Generate transactions only within the customer's active period
        active_days = (end_date - join_date).days
        if active_days <= 0:
            continue
            
        for _ in range(n_tx):
            # Generate transaction date within customer's active period
            days_offset = np.random.randint(0, active_days)
            txn_date = join_date + timedelta(days=days_offset)
            
            # Skip transactions after churn date
            if will_churn and txn_date >= churn_date:
                continue
                
            product = np.random.choice(available_products)
            
            # Make amount realistic based on product and customer profile
            if product == "loan":
                amount = np.round(np.random.normal(500, 100), 2)  # Loan payments
            elif product == "credit_card":
                amount = -np.round(np.random.exponential(scale=150), 2)  # Credit purchases (negative)
            elif product == "savings":
                amount = np.round(np.random.exponential(scale=200), 2)  # Deposits (positive)
            else:  # checking, debit
                amount = -np.round(np.random.exponential(scale=75), 2)  # Debits (negative)
            
            # Adjust amounts based on income bracket
            if row['income_bracket'] == 'high':
                amount *= 2
            elif row['income_bracket'] == 'low':
                amount *= 0.6

            txns.append({
                "customer_id": row["customer_id"],
                "product": product,
                "amount": amount,
                "txn_type": np.random.choice(['purchase', 'payment', 'deposit', 'withdrawal']),
                "timestamp": txn_date,
                "churned_customer": will_churn
            })
    
    return pd.DataFrame(txns)

def generate_interactions(customers, transactions, avg_int_per_customer=20):
    interactions = []
    
    # Get customer activity periods from transactions
    customer_periods = transactions.groupby('customer_id').agg({
        'timestamp': ['min', 'max'],
        'churned_customer': 'first'
    }).reset_index()
    customer_periods.columns = ['customer_id', 'first_tx', 'last_tx', 'churned_customer']
    
    for idx, row in tqdm(customers.iterrows(), total=len(customers), desc="Generating interactions"):
        # Get customer's transaction period
        customer_period = customer_periods[customer_periods['customer_id'] == row['customer_id']]
        
        if len(customer_period) == 0:
            # Customer has no transactions, use join date to a short period
            join_date = datetime.combine(row['join_date'], datetime.min.time()) if isinstance(row['join_date'], type(datetime.now().date())) else row['join_date']
            start_date = join_date
            end_date = min(join_date + timedelta(days=30), END_DATE)
            is_churned = False
        else:
            start_date = customer_period.iloc[0]['first_tx']
            end_date = customer_period.iloc[0]['last_tx']
            is_churned = customer_period.iloc[0]['churned_customer']
        
        # Adjust interaction volume based on customer profile
        int_multiplier = 1.0
        if row['age'] < 30:  # Younger customers interact more digitally
            int_multiplier = 1.3
        if row['has_credit_card']:
            int_multiplier += 0.2
        if is_churned:  # Churned customers had fewer interactions
            int_multiplier *= 0.7
            
        n_int = int(np.random.poisson(avg_int_per_customer * int_multiplier))
        
        # Build a weighted interaction profile
        weights = {
            'login': 1.0,
            'support_call': 0.1,
            'email_click': 0.2
        }

        # Adjust weights based on product ownership
        if row["has_credit_card"]:
            weights['support_call'] += 0.3
            weights['email_click'] += 0.2
        if row["has_loan"]:
            weights['support_call'] += 0.4
            weights['email_click'] += 0.1
        if row["has_savings"]:
            weights['email_click'] += 0.2
        if row["has_checking"]:
            weights['login'] += 0.5

        # Normalize to make it a probability distribution
        total_weight = sum(weights.values())
        interaction_types = list(weights.keys())
        probabilities = [w / total_weight for w in weights.values()]

        # Generate interactions within the customer's active period
        active_days = (end_date - start_date).days
        if active_days <= 0:
            active_days = 1
            
        for _ in range(n_int):
            interaction_date = start_date + timedelta(days=np.random.randint(0, active_days))
            
            interactions.append({
                "customer_id": row["customer_id"],
                "interaction_type": np.random.choice(interaction_types, p=probabilities),
                "timestamp": interaction_date
            })
    
    return pd.DataFrame(interactions)

def generate_churn_labels(customers, transactions):
    """Generate churn labels based on actual transaction patterns"""
    # Get the last transaction date for each customer
    latest_tx = transactions.groupby("customer_id").agg({
        'timestamp': 'max',
        'churned_customer': 'first'
    }).reset_index()
    latest_tx.columns = ["customer_id", "last_tx_date", "churned_customer"]
    
    # Merge with customer data
    merged = pd.merge(customers, latest_tx, on="customer_id", how="left")
    
    # Calculate days since last transaction
    merged["days_since_last_tx"] = (END_DATE - merged["last_tx_date"]).dt.days.fillna(9999)
    
    # Define churn: either marked as churned during simulation OR no activity for 90+ days
    merged["churned"] = ((merged["churned_customer"] == True) | 
                        (merged["days_since_last_tx"] > 90)).astype(int)
    
    return merged[["customer_id", "churned", "last_tx_date", "days_since_last_tx"]]

# Generate datasets
print("Starting data generation...")
customer_df = generate_customer_profiles(NUM_CUSTOMERS)
print(f"Generated {len(customer_df)} customer profiles")

transaction_df = generate_transactions(customer_df, avg_tx_per_customer=100)
print(f"Generated {len(transaction_df)} transactions")

interaction_df = generate_interactions(customer_df, transaction_df, avg_int_per_customer=20)
print(f"Generated {len(interaction_df)} interactions")

churn_df = generate_churn_labels(customer_df, transaction_df)
print(f"Generated churn labels - {churn_df['churned'].sum()} churned customers out of {len(churn_df)}")

# Data quality checks
print("\n=== DATA QUALITY CHECKS ===")
print(f"Transaction date range: {transaction_df['timestamp'].min()} to {transaction_df['timestamp'].max()}")
print(f"Interaction date range: {interaction_df['timestamp'].min()} to {interaction_df['timestamp'].max()}")
print(f"Transactions after END_DATE: {(transaction_df['timestamp'] > END_DATE).sum()}")
print(f"Interactions after END_DATE: {(interaction_df['timestamp'] > END_DATE).sum()}")
print(f"Churn rate: {churn_df['churned'].mean():.2%}")

# Remove the helper columns from dataframes for more realistic feat. engineering
transaction_df = transaction_df.drop('churned_customer', axis=1)
customer_df = customer_df.drop('income_bracket', axis=1)
churn_df = churn_df.drop(['last_tx_date', 'days_since_last_tx'], axis=1)

# Convert datetime columns to microsecond precision to be compatible with Spark
transaction_df['timestamp'] = transaction_df['timestamp'].astype('datetime64[us]')
interaction_df['timestamp'] = interaction_df['timestamp'].astype('datetime64[us]')

# Save as Parquet for PySpark
save_dir = Path(__file__).resolve().parent.parent / "raw"
save_dir.mkdir(exist_ok=True)

customer_df.to_parquet(save_dir / "customers.parquet", index=False)
transaction_df.to_parquet(save_dir / "transactions.parquet", index=False)
interaction_df.to_parquet(save_dir / "interactions.parquet", index=False)
churn_df.to_parquet(save_dir / "churn_labels.parquet", index=False)

print(f"\nDatasets saved to {save_dir}/")
print("Files created:")
print("- customers.parquet")
print("- transactions.parquet") 
print("- interactions.parquet")
print("- churn_labels.parquet")