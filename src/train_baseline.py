import pandas as pd

# Load your Kaggle data
df = pd.read_csv("data/cbb25.csv")

# Inspect the structure
print(df.head())
print(df.columns)
