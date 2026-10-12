import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Loading CSV

csv_path = glob.glob("data/*.csv")[0]
df = pd.read_csv(csv_path)
print("File:", csv_path)
print("Shape (rows, columns):", df.shape)
print(df.head())

# 2. Cleaning

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0])
df["TotalCharges"] = df["TotalCharges"].fillna(0)

df["Churn"] = (df["Churn"] == "Yes").astype(int)

# 2. (imbalanced check)
churn_rate = df["Churn"].mean()
print(f"\nChurn rate: {churn_rate:.1%}")

# 3. EDA charts
sns.set_theme(style="whitegrid")

# Chart 1: Contract type vs churn
plt.figure(figsize=(6, 4))
df.groupby("Contract")["Churn"].mean().plot(kind="bar", color="tomato")
plt.title("Churn rate by contract type")
plt.ylabel("Churn rate")
plt.tight_layout()
plt.savefig("reports/churn_by_contract.png")
plt.close()

# Chart 2: Tenure vs churn
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, multiple="stack")
plt.title("Tenure (months) by churn")
plt.tight_layout()
plt.savefig("reports/churn_by_tenure.png")
plt.close()

# Chart 3: Monthly charges vs churn
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly charges: stayed (0) vs churned (1)")
plt.tight_layout()
plt.savefig("reports/churn_by_charges.png")
plt.close()

# Chart 4: Internet service vs churn
plt.figure(figsize=(6, 4))
df.groupby("InternetService")["Churn"].mean().plot(kind="bar", color="steelblue")
plt.title("Churn rate by internet service")
plt.ylabel("Churn rate")
plt.tight_layout()
plt.savefig("reports/churn_by_internet.png")
plt.close()

# 5. Business cost matrix
COST_OFFER = 10       # retention offer budget
COST_LOST = 100
print(f"\nCost of offer: ${COST_OFFER} | Cost of losing customer: ${COST_LOST}")

# 6. Saved Cleaned data
df.to_csv("data/telco_clean.csv", index=False)
print("Done. Charts reports/ folder mein hain.")