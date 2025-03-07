import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure static/images directory exists
if not os.path.exists('static/images'):
    os.makedirs('static/images')

# Load the dataset
data = pd.read_csv('data.csv')

# Visualization 1: Age Distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['age'], bins=20, kde=True, color='blue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('static/images/age_distribution.png')
plt.close()

# Visualization 2: Age vs Cholesterol
plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='chol', hue='target', data=data, palette='coolwarm')
plt.title('Age vs Cholesterol by Heart Disease')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.savefig('static/images/age_vs_chol.png')
plt.close()

# Visualization 3: Sex vs Target
plt.figure(figsize=(8, 6))
sns.countplot(x='sex', hue='target', data=data, palette='Set2')
plt.title('Sex vs Heart Disease')
plt.xlabel('Sex (0: Female, 1: Male)')
plt.ylabel('Count')
plt.savefig('static/images/sex_vs_target.png')
plt.close()

print("Visualizations generated and saved in static/images/")