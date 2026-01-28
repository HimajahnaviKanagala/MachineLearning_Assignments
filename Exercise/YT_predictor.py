import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'ctr': [3.2, 5.8, 2.1, 7.4, 4.5, 6.2, 1.8, 5.1, 3.9, 8.5,
            4.2, 2.8, 6.8, 3.5, 7.1, 2.4, 5.5, 4.8, 6.5, 3.1,
            7.8, 2.6, 5.3, 4.1, 6.9, 3.7, 8.1, 2.2, 5.9, 4.6],

    'total_views': [12000, 28000, 8500, 42000, 19000, 33000, 7000, 24000, 16000, 51000,
                    18000, 11000, 38000, 14500, 44000, 9500, 26000, 21000, 35000, 13000,
                    47000, 10000, 25000, 17500, 40000, 15000, 49000, 8000, 31000, 20000]
}

df = pd.DataFrame(data)

# Explore the data
print("First 5 rows of data:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())

# visualize the data
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))
plt.scatter(df['ctr'], df['total_views'],
            c=df['ctr'], cmap='viridis', s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
plt.colorbar(label='CTR Intensity')
plt.xlabel('ctr',fontsize=12, fontweight='bold')
plt.ylabel('Total views', fontsize=12, fontweight='bold')
plt.title('CTR VS Total Views', fontsize=14, fontweight='bold',pad=15)
plt.tight_layout()
plt.show()

# prepare data for training
X=df[['ctr']]
y=df['total_views']

# split into training and testing sets
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

# train the model
model=LinearRegression()
model.fit(X_train, y_train)

# checking model parameters
print('\nModel Parameters')
print(f"Slope (coefficient): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# predictions on test data
y_pred=model.predict(X_test)

print("\nPredictions VS Actual:")
results=pd.DataFrame({'Actual':y_test.values, 'predicted':y_pred.round()})
print(results)

# visualize the regression line
plt.figure(figsize=(10, 6))
plt.scatter(df['ctr'], df['total_views'],
            c='#3498db', s=100, alpha=0.7, edgecolors='white', linewidth=2.5, label='Regression Line')
plt.xlabel('CTR', fontsize=12, fontweight='bold')
plt.ylabel('Total Views', fontsize=12, fontweight='bold')
plt.title('Linear Regression: CTR VS Total Views', fontsize=14, fontweight='bold', pad=15)
plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
plt.tight_layout()
plt.show()

# Answer arjun's question
new_ctr=8.0
predicted_views=model.predict([[new_ctr]])
print(f"\nArjun's Question: If I get {new_ctr} ctr, how many views?")
print(f"Predicted Total Views: {predicted_views[0]:,.0f}")
