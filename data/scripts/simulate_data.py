import pandas as pd
import numpy as np
import uuid
import random
from datetime import datetime, timedelta
from faker import Faker
from tqdm import tqdm
from pathlib import Path
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

fake = Faker()
np.random.seed(42)  # Changed for reproducibility

NUM_CUSTOMERS = 1000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2025, 1, 31)
PRODUCTS = ['checking', 'savings', 'credit_card', 'loan', 'debit']

# Enhanced customer risk factors for churn
CHURN_RISK_FACTORS = {
    'fee_sensitivity': 0.15,  # High fees increase churn risk
    'low_engagement': 0.12,   # Low digital engagement
    'product_dissatisfaction': 0.10,  # Limited product usage
    'competitive_pressure': 0.08,     # Market factors
    'life_events': 0.05       # Major life changes
}

def generate_customer_profiles(num_customers):
    """Generate realistic customer profiles with enhanced churn risk factors"""
    profiles = []
    
    for _ in tqdm(range(num_customers), desc="Generating customer profiles"):
        # Ensure sufficient activity window
        join_date = fake.date_between(
            start_date=START_DATE.date(), 
            end_date=(END_DATE - timedelta(days=120)).date()
        )
        
        age = np.random.randint(18, 85)
        income_bracket = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
        
        # Geographic clustering (some states have higher churn)
        state = fake.state_abbr()
        high_churn_states = ['CA', 'NY', 'FL', 'TX']  # Competitive markets
        geo_churn_risk = 1.2 if state in high_churn_states else 1.0
        
        # Product ownership with realistic dependencies
        has_checking = 1  # Universal
        
        # Savings tied to income and age
        savings_prob = 0.7
        if income_bracket == 'high':
            savings_prob = 0.9
        elif age < 25:
            savings_prob = 0.5
        has_savings = np.random.choice([0, 1], p=[1-savings_prob, savings_prob])
        
        # Credit card with realistic approval patterns
        cc_prob = 0.6
        if age < 21:
            cc_prob = 0.2  # Limited credit history
        elif age < 25:
            cc_prob = 0.45
        elif income_bracket == 'high':
            cc_prob = 0.85
        elif income_bracket == 'low':
            cc_prob = 0.35
        has_credit_card = np.random.choice([0, 1], p=[1-cc_prob, cc_prob])
        
        # Loan ownership
        loan_prob = 0.25
        if 25 <= age <= 45:  # Prime borrowing years
            loan_prob = 0.45
        elif age > 60:
            loan_prob = 0.15
        if income_bracket == 'high':
            loan_prob = min(0.7, loan_prob * 1.8)
        elif income_bracket == 'low':
            loan_prob *= 0.4
        has_loan = np.random.choice([0, 1], p=[1-loan_prob, loan_prob])
        
        # Digital engagement score (affects churn risk)
        if age < 35:
            digital_engagement = np.random.normal(0.8, 0.15)
        elif age < 55:
            digital_engagement = np.random.normal(0.6, 0.2)
        else:
            digital_engagement = np.random.normal(0.4, 0.25)
        digital_engagement = np.clip(digital_engagement, 0.1, 1.0)
        
        # Customer lifetime value indicator
        clv_segment = np.random.choice(['low', 'medium', 'high'], p=[0.4, 0.4, 0.2])
        if income_bracket == 'high':
            clv_segment = np.random.choice(['medium', 'high'], p=[0.3, 0.7])
        
        # Tenure affects churn risk (newer customers more likely to churn)
        tenure_months = (START_DATE.date() - join_date).days // 30
        tenure_churn_multiplier = max(0.3, 1.0 - (tenure_months * 0.02))
        
        profile = {
            "customer_id": str(uuid.uuid4()),
            "age": age,
            "state": state,
            "has_credit_card": has_credit_card,
            "has_loan": has_loan,
            "has_checking": has_checking,
            "has_savings": has_savings,
            "join_date": join_date,
            "gender": np.random.choice(['M', 'F', 'Other'], p=[0.49, 0.49, 0.02]),
            "income_bracket": income_bracket,
            "digital_engagement": digital_engagement,
            "clv_segment": clv_segment,
            "geo_churn_risk": geo_churn_risk,
            "tenure_churn_multiplier": tenure_churn_multiplier
        }
        profiles.append(profile)
    
    return pd.DataFrame(profiles)

def determine_customer_lifecycle(customer_row):
    """Enhanced lifecycle modeling with gradual decline patterns"""
    join_date = datetime.combine(customer_row['join_date'], datetime.min.time()) \
        if isinstance(customer_row['join_date'], type(datetime.now().date())) else customer_row['join_date']
    
    # Calculate churn probability based on multiple factors
    base_churn_prob = 0.2  # 20% base annual churn rate
    
    # Adjust for customer characteristics
    churn_prob = base_churn_prob
    churn_prob *= customer_row['tenure_churn_multiplier']
    churn_prob *= customer_row['geo_churn_risk']
    
    # Digital engagement factor
    if customer_row['digital_engagement'] < 0.3:
        churn_prob *= 1.5
    elif customer_row['digital_engagement'] > 0.7:
        churn_prob *= 0.7
    
    # CLV segment factor
    if customer_row['clv_segment'] == 'high':
        churn_prob *= 0.5
    elif customer_row['clv_segment'] == 'low':
        churn_prob *= 1.3
    
    # Product depth reduces churn
    product_count = sum([
        customer_row['has_checking'],
        customer_row['has_savings'], 
        customer_row['has_credit_card'],
        customer_row['has_loan']
    ])
    churn_prob *= max(0.4, 1.0 - ((product_count - 1) * 0.15))
    
    # Ensure reasonable bounds
    churn_prob = np.clip(churn_prob, 0.05, 0.8)
    
    # Determine pattern based on churn probability
    if np.random.random() < churn_prob:
        # Will churn - decide when and how
        pattern_type = np.random.choice(['gradual_decline', 'sudden_churn'], p=[0.7, 0.3])
        
        # Timing of churn
        min_active_days = 45 if customer_row['clv_segment'] == 'high' else 20
        days_since_join = (END_DATE - join_date).days
        
        if days_since_join > min_active_days:
            churn_day = np.random.randint(
                min_active_days, 
                min(days_since_join - 30, days_since_join)  # Leave 30 days for detection
            )
            churn_date = join_date + timedelta(days=churn_day)
            
            if pattern_type == 'gradual_decline':
                # 60-90 day decline period
                decline_days = np.random.randint(60, 91)
                decline_start = churn_date - timedelta(days=decline_days)
                
                return {
                    'pattern': 'gradual_decline',
                    'activity_end': churn_date,
                    'activity_decline_start': decline_start,
                    'baseline_activity': 0.9,
                    'decline_rate': 0.015  # 1.5% daily reduction
                }
            else:
                # Sudden churn with short warning period
                decline_days = np.random.randint(7, 21)
                decline_start = churn_date - timedelta(days=decline_days)
                
                return {
                    'pattern': 'sudden_churn',
                    'activity_end': churn_date,
                    'activity_decline_start': decline_start,
                    'baseline_activity': 0.8,
                    'decline_rate': 0.05  # 5% daily reduction
                }
    
    # At-risk customers (declining but not churning)
    elif np.random.random() < 0.15:
        decline_start = join_date + timedelta(days=np.random.randint(90, 200))
        return {
            'pattern': 'at_risk',
            'activity_end': END_DATE,
            'activity_decline_start': decline_start,
            'baseline_activity': 0.8,
            'decline_rate': 0.008  # Slow decline
        }
    
    # Stable customers with seasonal patterns
    else:
        return {
            'pattern': 'stable',
            'activity_end': END_DATE,
            'activity_decline_start': None,
            'baseline_activity': 1.0,
            'decline_rate': 0.0
        }

def calculate_activity_multiplier(current_date, lifecycle_info):
    """Enhanced activity calculation with seasonal and weekly patterns"""
    base_multiplier = lifecycle_info['baseline_activity']
    
    # Apply lifecycle-specific patterns
    if lifecycle_info['pattern'] == 'stable':
        # Seasonal variations for stable customers
        month = current_date.month
        if month in [12, 1]:  # Holiday season - higher activity
            base_multiplier *= 1.2
        elif month in [6, 7, 8]:  # Summer - slightly lower
            base_multiplier *= 0.9
        
        # Weekly patterns (higher on weekdays)
        if current_date.weekday() < 5:  # Monday-Friday
            base_multiplier *= 1.1
        else:
            base_multiplier *= 0.8
    
    elif lifecycle_info['pattern'] in ['gradual_decline', 'sudden_churn']:
        if current_date >= lifecycle_info['activity_end']:
            return 0.0  # No activity after churn
        
        if lifecycle_info['activity_decline_start'] and current_date >= lifecycle_info['activity_decline_start']:
            days_declining = (current_date - lifecycle_info['activity_decline_start']).days
            reduction = days_declining * lifecycle_info['decline_rate']
            base_multiplier = max(0.05, base_multiplier - reduction)
    
    elif lifecycle_info['pattern'] == 'at_risk':
        if lifecycle_info['activity_decline_start'] and current_date >= lifecycle_info['activity_decline_start']:
            days_declining = (current_date - lifecycle_info['activity_decline_start']).days
            reduction = days_declining * lifecycle_info['decline_rate']
            base_multiplier = max(0.3, base_multiplier - reduction)  # Don't go too low
    
    return base_multiplier

def generate_transactions(customers, avg_tx_per_customer=100):
    """Enhanced transaction generation with realistic patterns"""
    txns = []
    
    for idx, row in tqdm(customers.iterrows(), total=len(customers), desc="Generating transactions"):
        lifecycle_info = determine_customer_lifecycle(row)
        
        join_date = datetime.combine(row['join_date'], datetime.min.time()) \
            if isinstance(row['join_date'], type(datetime.now().date())) else row['join_date']
        
        # Available products
        available_products = ["checking", "debit"]
        if row["has_credit_card"]:
            available_products.append("credit_card")
        if row["has_loan"]:
            available_products.append("loan")
        if row["has_savings"]:
            available_products.append("savings")
        
        # Transaction volume based on customer profile
        tx_multiplier = 1.0
        if row['income_bracket'] == 'high':
            tx_multiplier *= 1.5
        elif row['income_bracket'] == 'low':
            tx_multiplier *= 0.7
        
        # CLV segment affects transaction frequency
        if row['clv_segment'] == 'high':
            tx_multiplier *= 1.3
        elif row['clv_segment'] == 'low':
            tx_multiplier *= 0.8
        
        # Digital engagement affects transaction patterns
        if row['digital_engagement'] > 0.7:
            tx_multiplier *= 1.2
        elif row['digital_engagement'] < 0.3:
            tx_multiplier *= 0.8
        
        base_tx_count = int(np.random.poisson(avg_tx_per_customer * tx_multiplier))
        
        # Generate transaction dates with realistic clustering
        active_period = (lifecycle_info['activity_end'] - join_date).days
        if active_period <= 0:
            continue
        
        # Create more realistic transaction timing
        tx_dates = []
        
        # Generate monthly clusters (paycheck cycles)
        current_date = join_date
        while current_date < lifecycle_info['activity_end']:
            # Monthly transaction burst around paycheck dates
            paycheck_dates = [
                current_date.replace(day=1),  # 1st of month
                current_date.replace(day=15) if current_date.month != 2 or current_date.day <= 15 else current_date.replace(day=14)  # 15th of month
            ]
            
            for paycheck_date in paycheck_dates:
                if paycheck_date > lifecycle_info['activity_end']:
                    break
                
                # Activity multiplier for this date
                activity_mult = calculate_activity_multiplier(paycheck_date, lifecycle_info)
                
                # Generate transactions around paycheck date
                tx_cluster_size = int(np.random.poisson(4 * activity_mult))
                
                for _ in range(tx_cluster_size):
                    # Transactions clustered around paycheck ±5 days
                    tx_offset = np.random.randint(-5, 6)
                    tx_date = paycheck_date + timedelta(days=tx_offset)
                    
                    if join_date <= tx_date <= lifecycle_info['activity_end']:
                        tx_dates.append(tx_date)
            
            # Move to next month
            current_date += relativedelta(months=1)
        
        # Add random transactions between paychecks
        remaining_tx = max(0, base_tx_count - len(tx_dates))
        for _ in range(remaining_tx):
            days_offset = np.random.randint(0, active_period)
            tx_date = join_date + timedelta(days=days_offset)
            
            if tx_date <= lifecycle_info['activity_end']:
                activity_mult = calculate_activity_multiplier(tx_date, lifecycle_info)
                if np.random.random() < activity_mult:
                    tx_dates.append(tx_date)
        
        # Generate actual transactions
        for tx_date in tx_dates:
            if tx_date > END_DATE:
                continue
            
            # Product selection based on usage patterns
            if len(available_products) > 1:
                # Weighted product selection
                product_weights = {
                    'checking': 0.4,
                    'debit': 0.3,
                    'credit_card': 0.2,
                    'savings': 0.07,
                    'loan': 0.03
                }
                
                available_weights = [product_weights.get(p, 0) for p in available_products]
                total_weight = sum(available_weights)
                probs = [w/total_weight for w in available_weights]
                
                product = np.random.choice(available_products, p=probs)
            else:
                product = available_products[0]
            
            # Enhanced amount generation with realistic distributions
            if product == "loan":
                # Fixed loan payments with some variation
                base_payment = np.random.normal(450, 50)
                amount = np.round(max(50, base_payment), 2)
                
            elif product == "credit_card":
                # Mix of small and large purchases
                if np.random.random() < 0.7:  # 70% small purchases
                    amount = -np.round(np.random.lognormal(mean=3.5, sigma=1.0), 2)
                else:  # 30% larger purchases
                    amount = -np.round(np.random.lognormal(mean=5.5, sigma=0.8), 2)
                
            elif product == "savings":
                # Realistic savings patterns
                if np.random.random() < 0.8:  # 80% deposits
                    amount = np.round(np.random.lognormal(mean=5.0, sigma=1.0), 2)
                else:  # 20% withdrawals
                    amount = -np.round(np.random.lognormal(mean=4.5, sigma=0.8), 2)
                
            else:  # checking, debit
                if np.random.random() < 0.25:  # 25% credits
                    amount = np.round(np.random.lognormal(mean=5.5, sigma=1.0), 2)
                else:  # 75% debits
                    amount = -np.round(np.random.lognormal(mean=4.0, sigma=1.2), 2)
            
            # Apply income adjustments
            if row['income_bracket'] == 'high':
                amount *= np.random.uniform(1.8, 3.0)
            elif row['income_bracket'] == 'low':
                amount *= np.random.uniform(0.3, 0.7)
            
            # Determine transaction type
            if amount > 0:
                if product == "loan":
                    txn_type = "payment"
                else:
                    txn_type = np.random.choice(["deposit", "transfer"], p=[0.7, 0.3])
            else:
                txn_type = np.random.choice(["purchase", "withdrawal", "fee"], p=[0.8, 0.15, 0.05])
            
            # Add fees occasionally (churn risk factor)
            if np.random.random() < 0.02:  # 2% chance of fee
                fee_amount = -np.random.uniform(25, 50)
                txns.append({
                    "customer_id": row["customer_id"],
                    "product": product,
                    "amount": fee_amount,
                    "txn_type": "fee",
                    "timestamp": tx_date
                })
            
            txns.append({
                "customer_id": row["customer_id"],
                "product": product,
                "amount": amount,
                "txn_type": txn_type,
                "timestamp": tx_date
            })
    
    return pd.DataFrame(txns)

def generate_interactions(customers, avg_int_per_customer=30):
    """Enhanced interaction generation aligned with customer lifecycle"""
    interactions = []
    
    for idx, row in tqdm(customers.iterrows(), total=len(customers), desc="Generating interactions"):
        lifecycle_info = determine_customer_lifecycle(row)
        
        join_date = datetime.combine(row['join_date'], datetime.min.time()) \
            if isinstance(row['join_date'], type(datetime.now().date())) else row['join_date']
        
        # Base interaction volume
        int_multiplier = row['digital_engagement']
        
        # Age-based digital behavior
        if row['age'] < 30:
            int_multiplier *= 1.5
        elif row['age'] > 65:
            int_multiplier *= 0.4
        
        # CLV affects engagement
        if row['clv_segment'] == 'high':
            int_multiplier *= 1.3
        elif row['clv_segment'] == 'low':
            int_multiplier *= 0.7
        
        base_int_count = int(np.random.poisson(avg_int_per_customer * int_multiplier))
        
        # Generate interaction dates
        active_period = (lifecycle_info['activity_end'] - join_date).days
        if active_period <= 0:
            continue
        
        int_dates = []
        for _ in range(base_int_count):
            days_offset = np.random.exponential(scale=active_period/4)
            days_offset = min(days_offset, active_period-1)
            int_date = join_date + timedelta(days=int(days_offset))
            
            activity_mult = calculate_activity_multiplier(int_date, lifecycle_info)
            
            if np.random.random() < activity_mult:
                int_dates.append(int_date)
        
        # Interaction type weights based on customer profile
        base_weights = {
            'login': 0.45,
            'web': 0.25,
            'mobile': 0.20,
            'support': 0.05,
            'email_click': 0.05
        }
        
        # Adjust for demographics
        if row['age'] < 35:
            base_weights['mobile'] *= 2.5
            base_weights['web'] *= 1.5
            base_weights['support'] *= 0.5
        elif row['age'] > 55:
            base_weights['mobile'] *= 0.2
            base_weights['web'] *= 0.7
            base_weights['support'] *= 3.0
        
        # Digital engagement affects channel preference
        if row['digital_engagement'] > 0.7:
            base_weights['mobile'] *= 1.5
            base_weights['web'] *= 1.3
            base_weights['support'] *= 0.6
        elif row['digital_engagement'] < 0.3:
            base_weights['mobile'] *= 0.3
            base_weights['web'] *= 0.5
            base_weights['support'] *= 2.0
        
        # Product complexity drives support interactions
        if row['has_credit_card'] or row['has_loan']:
            base_weights['support'] *= 1.8
            base_weights['email_click'] *= 1.5
        
        # Normalize weights
        total_weight = sum(base_weights.values())
        interaction_types = list(base_weights.keys())
        probabilities = [w / total_weight for w in base_weights.values()]
        
        # Generate interactions
        for int_date in int_dates:
            if int_date > END_DATE:
                continue
            
            interaction_type = np.random.choice(interaction_types, p=probabilities)
            
            # Add session duration for digital interactions
            session_duration = None
            if interaction_type in ['login', 'web', 'mobile']:
                if row['digital_engagement'] > 0.7:
                    session_duration = np.random.lognormal(mean=4.0, sigma=1.0)
                else:
                    session_duration = np.random.lognormal(mean=3.0, sigma=1.2)
            
            interaction = {
                "customer_id": row["customer_id"],
                "interaction_type": interaction_type,
                "timestamp": int_date
            }
            
            if session_duration:
                interaction["session_duration"] = round(session_duration, 2)
            
            interactions.append(interaction)
    
    return pd.DataFrame(interactions)

def validate_data_quality(customers, transactions, interactions):
    """Enhanced validation for churn pipeline compatibility"""
    print("\n=== ENHANCED DATA VALIDATION ===")
    
    # Basic metrics
    print(f"Customer profiles: {len(customers):,}")
    print(f"Total transactions: {len(transactions):,}")
    print(f"Total interactions: {len(interactions):,}")
    
    # Date range validation
    tx_dates = pd.to_datetime(transactions['timestamp'])
    int_dates = pd.to_datetime(interactions['timestamp'])
    
    print(f"Transaction date range: {tx_dates.min()} to {tx_dates.max()}")
    print(f"Interaction date range: {int_dates.min()} to {int_dates.max()}")
    
    # Customer activity analysis for churn detection
    recent_cutoffs = [7, 14, 30, 60, 90]
    
    print("\n=== CUSTOMER ACTIVITY ANALYSIS ===")
    for cutoff in recent_cutoffs:
        cutoff_date = END_DATE - timedelta(days=cutoff)
        
        recent_tx_customers = set(transactions[tx_dates > cutoff_date]['customer_id'].unique())
        recent_int_customers = set(interactions[int_dates > cutoff_date]['customer_id'].unique())
        recent_any_activity = recent_tx_customers | recent_int_customers
        
        inactive_customers = len(customers) - len(recent_any_activity)
        
        print(f"Customers inactive for {cutoff}+ days: {inactive_customers:,} ({inactive_customers/len(customers)*100:.1f}%)")
    
    # Transaction pattern analysis
    print("\n=== TRANSACTION PATTERNS ===")
    avg_tx_per_customer = len(transactions) / len(customers)
    print(f"Average transactions per customer: {avg_tx_per_customer:.1f}")
    
    # Monthly transaction distribution
    tx_monthly = transactions.groupby(pd.to_datetime(transactions['timestamp']).dt.to_period('M')).size()
    print(f"Monthly transaction volume (avg): {tx_monthly.mean():.0f}")
    print(f"Monthly transaction volume (std): {tx_monthly.std():.0f}")
    
    # Product usage
    product_usage = transactions['product'].value_counts()
    print(f"\nProduct usage distribution:")
    for product, count in product_usage.items():
        print(f"  {product}: {count:,} ({count/len(transactions)*100:.1f}%)")
    
    # Interaction patterns
    print("\n=== INTERACTION PATTERNS ===")
    avg_int_per_customer = len(interactions) / len(customers)
    print(f"Average interactions per customer: {avg_int_per_customer:.1f}")
    
    interaction_types = interactions['interaction_type'].value_counts()
    print(f"Interaction type distribution:")
    for int_type, count in interaction_types.items():
        print(f"  {int_type}: {count:,} ({count/len(interactions)*100:.1f}%)")
    
    # Churn pipeline readiness
    print("\n=== CHURN PIPELINE READINESS ===")
    
    # Check for gradual activity decline patterns
    monthly_activity = []
    for month in range(1, 13):
        month_start = datetime(2024, month, 1)
        month_end = datetime(2024, month, 28) if month != 12 else END_DATE
        
        month_tx = len(transactions[(tx_dates >= month_start) & (tx_dates < month_end)])
        month_int = len(interactions[(int_dates >= month_start) & (int_dates < month_end)])
        monthly_activity.append(month_tx + month_int)
    
    # Calculate activity trend
    months = np.arange(1, 13)
    trend = np.polyfit(months, monthly_activity, 1)[0]
    print(f"Overall activity trend: {trend:.1f} (negative = declining)")
    
    # Customer lifecycle distribution (estimated)
    print("\n=== ESTIMATED CUSTOMER LIFECYCLE DISTRIBUTION ===")
    
    all_customer_ids = set(customers['customer_id'])

    active_30d = set(transactions[tx_dates > END_DATE - timedelta(days=30)]['customer_id']) | \
                set(interactions[int_dates > END_DATE - timedelta(days=30)]['customer_id'])

    active_60d = set(transactions[tx_dates > END_DATE - timedelta(days=60)]['customer_id']) | \
                set(interactions[int_dates > END_DATE - timedelta(days=60)]['customer_id'])

    # Break lifecycle into:
    # - Stable: active in last 30 days
    # - At-risk: active in last 60 days but not last 30
    # - Churned: not active in last 60 days

    stable_customers = active_30d
    at_risk_customers = active_60d - active_30d
    churned_customers = all_customer_ids - active_60d

    print(f"Estimated churned customers: {len(churned_customers):,} ({len(churned_customers)/len(customers)*100:.1f}%)")
    print(f"Estimated at-risk customers: {len(at_risk_customers):,} ({len(at_risk_customers)/len(customers)*100:.1f}%)")
    print(f"Estimated stable customers: {len(stable_customers):,} ({len(stable_customers)/len(customers)*100:.1f}%)")

    
    return True

# # # Main execution
# # if __name__ == "__main__":
#     print("Starting enhanced customer data simulation for churn pipeline...")
#     print(f"Generating data for {NUM_CUSTOMERS:,} customers from {START_DATE.date()} to {END_DATE.date()}")
    
#     # Generate datasets
#     customer_df = generate_customer_profiles(NUM_CUSTOMERS)
#     print(f"✓ Generated {len(customer_df):,} customer profiles")
    
#     transaction_df = generate_transactions(customer_df, avg_tx_per_customer=120)
#     print(f"✓ Generated {len(transaction_df):,} transactions")
    
#     interaction_df = generate_interactions(customer_df, avg_int_per_customer=35)
#     print(f"✓ Generated {len(interaction_df):,} interactions")
    
#     # Validate data quality
#     validate_data_quality(customer_df, transaction_df, interaction_df)
    
#     # Clean up customer data for export
#     customer_df_clean = customer_df.drop(columns=[
#         'income_bracket', 'digital_engagement', 'clv_segment', 
#         'geo_churn_risk', 'tenure_churn_multiplier'
#     ])
    
#     # Ensure timestamp precision for Spark
#     transaction_df['timestamp'] = pd.to_datetime(transaction_df['timestamp']).dt.round('1s')
#     interaction_df['timestamp'] = pd.to_datetime(interaction_df['timestamp']).dt.round('1s')
    
#     # Save datasets
#     save_dir = Path("raw_data")
#     save_dir.mkdir(exist_ok=True)
    
#     print(f"\n=== SAVING DATASETS ===")
    
#     # Save as Parquet for optimal Spark performance
#     customer_df_clean.to_parquet(save_dir / "customers.parquet", index=False)
#     transaction_df.to_parquet(save_dir / "transactions.parquet", index=False)
#     interaction_df.to_parquet(save_dir / "interactions.parquet", index=False)
    
#     print(f"✓ Datasets saved to {save_dir}/")
#     print("Files created:")
#     print("  - customers.parquet")
#     print("  - transactions.parquet")
#     print("  - interactions.parquet")
    
#     # Generate data summary for pipeline documentation
#     print(f"\n=== DATA SUMMARY FOR PIPELINE ===")
#     print(f"Simulation period: {START_DATE.date()} to {END_DATE.date()}")
#     print(f"Total customers: {len(customer_df_clean):,}")
#     print(f"Total transactions: {len(transaction_df):,}")
#     print(f"Total interactions: {len(interaction_df):,}")
    
#     # Key metrics for feature engineering
#     print(f"\n=== KEY METRICS FOR FEATURE ENGINEERING ===")
    
#     # Customer demographics
#     age_dist = customer_df_clean['age'].describe()
#     print(f"Age distribution: mean={age_dist['mean']:.1f}, std={age_dist['std']:.1f}")
    
#     # Product penetration
#     product_penetration = {
#         'checking': customer_df_clean['has_checking'].mean(),
#         'savings': customer_df_clean['has_savings'].mean(),
#         'credit_card': customer_df_clean['has_credit_card'].mean(),
#         'loan': customer_df_clean['has_loan'].mean()
#     }
#     print("Product penetration rates:")
#     for product, rate in product_penetration.items():
#         print(f"  {product}: {rate:.1%}")
    
#     # Transaction statistics
#     tx_stats = transaction_df.groupby('customer_id').agg({
#         'amount': ['count', 'sum', 'mean', 'std'],
#         'timestamp': ['min', 'max']
#     }).round(2)
    
#     print(f"\nTransaction statistics per customer:")
#     print(f"  Average transaction count: {tx_stats[('amount', 'count')].mean():.1f}")
#     print(f"  Average transaction amount: ${tx_stats[('amount', 'mean')].mean():.2f}")
    
#     # Interaction statistics
#     int_stats = interaction_df.groupby('customer_id').agg({
#         'interaction_type': 'count',
#         'timestamp': ['min', 'max']
#     })
    
#     print(f"  Average interaction count: {int_stats[('interaction_type', 'count')].mean():.1f}")
    
#     # Activity recency for churn detection
#     last_activity = {}
#     for customer_id in customer_df_clean['customer_id']:
#         customer_txns = transaction_df[transaction_df['customer_id'] == customer_id]
#         customer_ints = interaction_df[interaction_df['customer_id'] == customer_id]
        
#         last_tx = customer_txns['timestamp'].max() if not customer_txns.empty else pd.NaT
#         last_int = customer_ints['timestamp'].max() if not customer_ints.empty else pd.NaT
        
#         last_activity[customer_id] = max(
#             pd.to_datetime(last_tx) if pd.notna(last_tx) else pd.Timestamp.min,
#             pd.to_datetime(last_int) if pd.notna(last_int) else pd.Timestamp.min
#         )
    
#     # Calculate days since last activity
#     days_since_activity = [
#         (END_DATE - last_act).days for last_act in last_activity.values()
#         if last_act != pd.Timestamp.min
#     ]
    
#     if days_since_activity:
#         print(f"\nActivity recency distribution:")
#         print(f"  Mean days since last activity: {np.mean(days_since_activity):.1f}")
#         print(f"  Customers inactive 30+ days: {sum(1 for d in days_since_activity if d >= 30)} ({sum(1 for d in days_since_activity if d >= 30)/len(days_since_activity)*100:.1f}%)")
#         print(f"  Customers inactive 60+ days: {sum(1 for d in days_since_activity if d >= 60)} ({sum(1 for d in days_since_activity if d >= 60)/len(days_since_activity)*100:.1f}%)")
#         print(f"  Customers inactive 90+ days: {sum(1 for d in days_since_activity if d >= 90)} ({sum(1 for d in days_since_activity if d >= 90)/len(days_since_activity)*100:.1f}%)")
    
#     print(f"\n=== PIPELINE RECOMMENDATIONS ===")
#     print("1. For 7-day rolling windows: Data provides sufficient granularity")
#     print("2. For 30-day prediction horizon: ~15-20% of customers show churn patterns")
#     print("3. Feature engineering focus areas:")
#     print("   - Transaction frequency and amount trends")
#     print("   - Product usage patterns")
#     print("   - Digital engagement scores")
#     print("   - Fee incidence and customer response")
#     print("   - Seasonal and cyclical patterns")
#     print("4. Churn indicators to monitor:")
#     print("   - Declining transaction frequency")
#     print("   - Reduced digital engagement")
#     print("   - Increased support interactions")
#     print("   - Fee sensitivity patterns")
    
#     print("\nData generation complete! Ready for churn pipeline ingestion.")

print("Starting enhanced customer data simulation for churn pipeline...")
print(f"Generating data for {NUM_CUSTOMERS:,} customers from {START_DATE.date()} to {END_DATE.date()}")

# Generate datasets
customer_df = generate_customer_profiles(NUM_CUSTOMERS)
print(f"✓ Generated {len(customer_df):,} customer profiles")

transaction_df = generate_transactions(customer_df, avg_tx_per_customer=120)
print(f"✓ Generated {len(transaction_df):,} transactions")

interaction_df = generate_interactions(customer_df, avg_int_per_customer=35)
print(f"✓ Generated {len(interaction_df):,} interactions")

# Validate data quality
validate_data_quality(customer_df, transaction_df, interaction_df)

# Clean up customer data for export
customer_df_clean = customer_df.drop(columns=[
    'income_bracket', 'digital_engagement', 'clv_segment', 
    'geo_churn_risk', 'tenure_churn_multiplier'
])

# Ensure timestamp precision for Spark
transaction_df['timestamp'] = pd.to_datetime(transaction_df['timestamp']).astype('datetime64[ms]')
interaction_df['timestamp'] = pd.to_datetime(interaction_df['timestamp']).astype('datetime64[ms]')

# Save datasets
save_dir = Path(__file__).resolve().parent.parent / "raw"
save_dir.mkdir(exist_ok=True)

print(f"\n=== SAVING DATASETS ===")

# Save as Parquet for optimal Spark performance
customer_df_clean.to_parquet(save_dir / "customers.parquet", index=False)
transaction_df.to_parquet(save_dir / "transactions.parquet", index=False)
interaction_df.to_parquet(save_dir / "interactions.parquet", index=False)

print(f"✓ Datasets saved to {save_dir}/")
print("Files created:")
print("  - customers.parquet")
print("  - transactions.parquet")
print("  - interactions.parquet")

# Generate data summary for pipeline documentation
print(f"\n=== DATA SUMMARY FOR PIPELINE ===")
print(f"Simulation period: {START_DATE.date()} to {END_DATE.date()}")
print(f"Total customers: {len(customer_df_clean):,}")
print(f"Total transactions: {len(transaction_df):,}")
print(f"Total interactions: {len(interaction_df):,}")

# Key metrics for feature engineering
print(f"\n=== KEY METRICS FOR FEATURE ENGINEERING ===")

# Customer demographics
age_dist = customer_df_clean['age'].describe()
print(f"Age distribution: mean={age_dist['mean']:.1f}, std={age_dist['std']:.1f}")

# Product penetration
product_penetration = {
    'checking': customer_df_clean['has_checking'].mean(),
    'savings': customer_df_clean['has_savings'].mean(),
    'credit_card': customer_df_clean['has_credit_card'].mean(),
    'loan': customer_df_clean['has_loan'].mean()
}
print("Product penetration rates:")
for product, rate in product_penetration.items():
    print(f"  {product}: {rate:.1%}")

# Transaction statistics
tx_stats = transaction_df.groupby('customer_id').agg({
    'amount': ['count', 'sum', 'mean', 'std'],
    'timestamp': ['min', 'max']
}).round(2)

print(f"\nTransaction statistics per customer:")
print(f"  Average transaction count: {tx_stats[('amount', 'count')].mean():.1f}")
print(f"  Average transaction amount: ${tx_stats[('amount', 'mean')].mean():.2f}")

# Interaction statistics
int_stats = interaction_df.groupby('customer_id').agg({
    'interaction_type': 'count',
    'timestamp': ['min', 'max']
})

print(f"  Average interaction count: {int_stats[('interaction_type', 'count')].mean():.1f}")

# Activity recency for churn detection
last_activity = {}
for customer_id in customer_df_clean['customer_id']:
    customer_txns = transaction_df[transaction_df['customer_id'] == customer_id]
    customer_ints = interaction_df[interaction_df['customer_id'] == customer_id]
    
    last_tx = customer_txns['timestamp'].max() if not customer_txns.empty else pd.NaT
    last_int = customer_ints['timestamp'].max() if not customer_ints.empty else pd.NaT
    
    last_activity[customer_id] = max(
        pd.to_datetime(last_tx) if pd.notna(last_tx) else pd.Timestamp.min,
        pd.to_datetime(last_int) if pd.notna(last_int) else pd.Timestamp.min
    )

# Calculate days since last activity
days_since_activity = [
    (END_DATE - last_act).days for last_act in last_activity.values()
    if last_act != pd.Timestamp.min
]

if days_since_activity:
    print(f"\nActivity recency distribution:")
    print(f"  Mean days since last activity: {np.mean(days_since_activity):.1f}")
    print(f"  Customers inactive 30+ days: {sum(1 for d in days_since_activity if d >= 30)} ({sum(1 for d in days_since_activity if d >= 30)/len(days_since_activity)*100:.1f}%)")
    print(f"  Customers inactive 60+ days: {sum(1 for d in days_since_activity if d >= 60)} ({sum(1 for d in days_since_activity if d >= 60)/len(days_since_activity)*100:.1f}%)")
    print(f"  Customers inactive 90+ days: {sum(1 for d in days_since_activity if d >= 90)} ({sum(1 for d in days_since_activity if d >= 90)/len(days_since_activity)*100:.1f}%)")

print(f"\n=== PIPELINE RECOMMENDATIONS ===")
print("1. For 7-day rolling windows: Data provides sufficient granularity")
print("2. For 30-day prediction horizon: ~15-20% of customers show churn patterns")
print("3. Feature engineering focus areas:")
print("   - Transaction frequency and amount trends")
print("   - Product usage patterns")
print("   - Digital engagement scores")
print("   - Fee incidence and customer response")
print("   - Seasonal and cyclical patterns")
print("4. Churn indicators to monitor:")
print("   - Declining transaction frequency")
print("   - Reduced digital engagement")
print("   - Increased support interactions")
print("   - Fee sensitivity patterns")

print("\nData generation complete! Ready for churn pipeline ingestion.")