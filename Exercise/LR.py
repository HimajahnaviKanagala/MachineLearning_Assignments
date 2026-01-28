# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# creating the data
data = {
    'first_hour_likes': [120, 340, 89, 510, 230, 670, 45, 390, 150, 720,
                         280, 95, 440, 180, 560, 75, 320, 480, 210, 630,
                         110, 350, 85, 410, 260, 590, 140, 470, 55, 380],
    
    'total_views': [8500, 22000, 5200, 41000, 15000, 53000, 3200, 28000, 10500, 58000,
                    18500, 6800, 32000, 12000, 44000, 4500, 21000, 36000, 14000, 49000,
                    7800, 23500, 5000, 30000, 17000, 46000, 9500, 35000, 3800, 27000]
}
df=pd.DataFrame(data)

# Explore the data
print("First 5 rows of data:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())

# visualize the data
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))
plt.scatter(df['first_hour_likes'], df['total_views'],
            c=df['first_hour_likes'], cmap='viridis', s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
plt.colorbar(label='Likes Intensity')
plt.xlabel('First Hour Likes',fontsize=12, fontweight='bold')
plt.ylabel('Total views', fontsize=12, fontweight='bold')
plt.title('First Hour Likes VS Total Views', fontsize=14, fontweight='bold',pad=15)
plt.tight_layout()
plt.show()

# prepare data for training
X=df[['first_hour_likes']]
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
plt.scatter(df['first_hour_likes'], df['total_views'],
            c='#3498db', s=100, alpha=0.7, edgecolors='white', linewidth=2.5, label='Regression Line')
plt.xlabel('First Hour Likes', fontsize=12, fontweight='bold')
plt.ylabel('Total Views', fontsize=12, fontweight='bold')
plt.title('Linear Regression: First Hour Likes VS Total Views', fontsize=14, fontweight='bold', pad=15)
plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
plt.tight_layout()
plt.show()

# Answer Raj's question
new_likes=750
predicted_views=model.predict([[new_likes]])
print(f"\nRaj's Question: If I get {new_likes} first-hour likes, how many views?")
print(f"Predicted Total Views: {predicted_views[0]:,.0f}")
