# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris

# Load the Iris dataset
try:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("Dataset loaded successfully!\n")
except Exception as e:
    print("Error loading dataset:", e)

# Show the first few rows
print("First 5 rows:")
print(df.head())

# Check data types and missing values
print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

# Clean the dataset (no missing values in Iris, but we'll keep the step)
df = df.dropna()

# Basic statistics
print("\nBasic statistics:")
print(df.describe())

# Group by species and compute mean
grouped_means = df.groupby('species').mean()
print("\nMean values grouped by species:")
print(grouped_means)

# Observations
print("\nObservations:")
print("Setosa has the smallest average petal size while Virginica has the largest.")
print("Versicolor falls between the two in all dimensions.")

# -------------------
# Data Visualizations
# -------------------

# Set style
sns.set(style="whitegrid")

# 1. Line chart - We'll fake a 'time' column for plotting trend-like data
df['index'] = df.index
plt.figure(figsize=(10, 5))
plt.plot(df['index'], df['sepal length (cm)'], label='Sepal Length')
plt.plot(df['index'], df['petal length (cm)'], label='Petal Length')
plt.title("Trend of Sepal and Petal Length Over Index (as time)")
plt.xlabel("Index (simulated time)")
plt.ylabel("Length (cm)")
plt.legend()
plt.show()

# 2. Bar chart - Average petal length by species
plt.figure(figsize=(8, 5))
sns.barplot(x='species', y='petal length (cm)', data=df)
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# 3. Histogram - Distribution of sepal width
plt.figure(figsize=(8, 5))
plt.hist(df['sepal width (cm)'], bins=10, color='skyblue', edgecolor='black')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot - Sepal length vs Petal length
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.show()
