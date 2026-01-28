import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Dataset
data = {
    'ram_gb': [4, 8, 4, 16, 8, 8, 4, 16, 8, 16,
               4, 8, 4, 16, 8, 8, 4, 16, 8, 16,
               4, 8, 4, 16, 8, 8, 4, 16, 8, 16],

    'storage_gb': [256, 512, 128, 512, 256, 512, 256, 1024, 256, 512,
                   128, 512, 256, 1024, 256, 512, 128, 512, 256, 1024,
                   256, 512, 128, 512, 256, 512, 256, 1024, 256, 512],

    'processor_ghz': [2.1, 2.8, 1.8, 3.2, 2.4, 3.0, 2.0, 3.5, 2.6, 3.0,
                      1.6, 2.8, 2.2, 3.4, 2.5, 2.9, 1.9, 3.1, 2.3, 3.6,
                      2.0, 2.7, 1.7, 3.3, 2.4, 3.0, 2.1, 3.5, 2.6, 3.2],

    'price_inr': [28000, 45000, 22000, 72000, 38000, 52000, 26000, 95000, 42000, 68000,
                  20000, 48000, 29000, 88000, 40000, 50000, 23000, 70000, 36000, 98000,
                  25000, 46000, 21000, 75000, 39000, 53000, 27000, 92000, 43000, 73000]
}

df = pd.DataFrame(data)

#explore the data
print("First 5 rows of data:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())

#visualize the data
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes=plt.subplots(1, 3, figsize=(18, 6))

scatter1=axes[0].scatter(df['ram_gb'], df['price_inr'],
                         c=df['price_inr'], cmap='viridis', s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
axes[0].set_xlabel('Ram (GB)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Price (INR)', fontsize=12, fontweight='bold')
axes[0].set_title('Ram VS Price', fontsize=14, fontweight='bold', pad=15)
plt.colorbar(scatter1, ax=axes[0], label='Price Intensity')

scatter2 = axes[1].scatter(df['storage_gb'], df['price_inr'],
                           c=df['price_inr'], cmap='plasma', s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
axes[1].set_xlabel('Storage (GB)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Price (INR)', fontsize=12, fontweight='bold')
axes[1].set_title('Storage VS Price', fontsize=14, fontweight='bold', pad=15)
plt.colorbar(scatter2, ax=axes[1], label='Price Intensity')

scatter3 = axes[2].scatter(df['processor_ghz'], df['price_inr'],
                           c=df['price_inr'], cmap='inferno', s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
axes[2].set_xlabel('Processor (GHz)', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Price (INR)', fontsize=12, fontweight='bold')
axes[2].set_title('Processor VS Price', fontsize=14, fontweight='bold', pad=15)
plt.colorbar(scatter3, ax=axes[2], label='Price Intensity')

plt.tight_layout()
plt.show()

# preapre data for training
X=df[['ram_gb', 'storage_gb', 'processor_ghz']]
y=df['price_inr']

# split into train and test sets
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=42)

# create and train the model
model=LinearRegression()
model.fit(X_train,y_train)

# check model parameters
print("\nModel Parameters:")
print(f"Coefficient: Ram={model.coef_[0]:.2f}, Storage={model.coef_[1]:.2f}, Processor={model.coef_[2]:.2f}")
print(f"Intercept:{model.intercept_:.2f}")

# predictions on test data
y_pred=model.predict(X_test)

print("\nPredictions VS Actual:")
results=pd.DataFrame({'Actual':y_test.values, 'Predicted':y_pred.round()})
print(results)

# visualize actual vs predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle='--')
plt.xlabel('Actual Price (INR)', fontweight='bold')
plt.ylabel('Predicted Price (INR)', fontweight='bold')
plt.title('Actual vs Predicted Laptop Prices', fontweight='bold')
plt.tight_layout()
plt.show()

# Model Accuracy
score=r2_score(y_test, y_pred)
print(f"\nModel Accuracy (R2 Score):{score:.2f}")

# Answer meera's question
new_price=[[16, 512, 3.2]]
predicted_rate=model.predict(new_price)
print(f"\nMeera's Question : 16GB RAM, 512GB Storage, 3.2 GHz processor ")
print(f"Predicted Laptop Price: ₹{predicted_rate[0]:,.0f}")





