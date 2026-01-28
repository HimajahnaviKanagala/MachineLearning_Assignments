import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'distance_km': [2.5, 6.0, 1.2, 8.5, 3.8, 5.2, 1.8, 7.0, 4.5, 9.2,
                    2.0, 6.5, 3.2, 7.8, 4.0, 5.8, 1.5, 8.0, 3.5, 6.8,
                    2.2, 5.5, 4.2, 9.0, 2.8, 7.2, 3.0, 6.2, 4.8, 8.2],

    'prep_time_min': [10, 20, 8, 25, 12, 18, 7, 22, 15, 28,
                      9, 19, 11, 24, 14, 17, 6, 26, 13, 21,
                      10, 16, 14, 27, 11, 23, 12, 18, 15, 25],

    'delivery_time_min': [18, 38, 12, 52, 24, 34, 14, 45, 29, 58,
                          15, 40, 21, 50, 27, 35, 11, 54, 23, 43,
                          17, 32, 26, 56, 19, 47, 20, 37, 30, 53]
}

df = pd.DataFrame(data)

#explore the data
print("First 5 rows of data:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())

#visualize the data
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes=plt.subplots(1, 2, figsize=(14, 6))

scatter1=axes[0].scatter(df['distance_km'], df['delivery_time_min'],
                         c=df['delivery_time_min'], cmap='viridis', s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
axes[0].set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Delivery Time (min)', fontsize=12, fontweight='bold')
axes[0].set_title('Distance VS Delivery Time', fontsize=14, fontweight='bold', pad=15)
plt.colorbar(scatter1, ax=axes[0], label='Delivery Intensity')

scatter2 = axes[1].scatter(df['prep_time_min'], df['delivery_time_min'],
                           c=df['delivery_time_min'], cmap='plasma', s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
axes[1].set_xlabel('Preparation Time (min)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Delivery Time (min)', fontsize=12, fontweight='bold')
axes[1].set_title('Preparation VS Delivery Time', fontsize=14, fontweight='bold', pad=15)
plt.colorbar(scatter2, ax=axes[1], label='Delivery Intensity')

plt.tight_layout()
plt.show()

# preapre data for training
X=df[['distance_km', 'prep_time_min']]
y=df['delivery_time_min']

# split into train and test sets
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=42)

# create and train the model
model=LinearRegression()
model.fit(X_train,y_train)

# check model parameters
print("\nModel Parameters:")
print(f"Coefficient: Distance={model.coef_[0]:.2f}, Preparation={model.coef_[1]:.2f}")
print(f"Intercept:{model.intercept_:.2f}")

# predictions on test data
y_pred=model.predict(X_test)

print("\nPredictions VS Actual:")
results=pd.DataFrame({'Actual':y_test.values, 'Predicted':y_pred.round()})
print(results)

# visualize actual vs predicted delivery time
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle='--')
plt.xlabel('Actual Delivery Time (min)', fontweight='bold')
plt.ylabel('Predicted Delivery Time (min)', fontweight='bold')
plt.title('Actual vs Predicted Delivery Time', fontweight='bold')
plt.tight_layout()
plt.show()


# Model Accuracy
from sklearn.metrics import r2_score
score=r2_score(y_test, y_pred)
print(f"\nModel Accuracy (R2 Score):{score:.2f}")

# Answer vikram's question
new_order=[[7, 15]]
predicted_rate=model.predict(new_order)
print(f"\nVikram's Question : 7km distance, 15min preparation ")
print(f"Predicted Delivery Time: {predicted_rate[0]:,.0f}min")




