import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python pair_plot.py <dataset.csv>")
    sys.exit(1)

filename = sys.argv[1]

df = pd.read_csv(filename)

# Select only numerical columns
numerical_cols = df.select_dtypes(include='number').columns

# Drop rows with missing values in numerical columns or house
df_clean = df.dropna(subset=list(numerical_cols) + ['Hogwarts House'])

# Make pair plot colored by house
pd.plotting.scatter_matrix(
    df_clean[numerical_cols],
    figsize=(12, 12),
    diagonal='hist',
    c=df_clean['Hogwarts House'].astype('category').cat.codes,
    cmap='tab10'
)
plt.suptitle("Pair Plot of Numerical Features by House")
plt.show()