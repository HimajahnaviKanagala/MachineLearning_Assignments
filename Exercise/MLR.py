import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#data
data = {
    'pages': [5, 12, 3, 8, 15, 4, 10, 6, 20, 7,
              9, 2, 14, 5, 11, 3, 8, 18, 6, 13,
              4, 16, 7, 10, 2, 12, 5, 9, 15, 6],

    'deadline_days': [14, 7, 21, 5, 10, 18, 6, 12, 8, 15,
                      4, 25, 9, 11, 7, 20, 6, 5, 14, 8,
                      16, 6, 10, 4, 22, 7, 13, 5, 9, 11],

    'rate_inr': [8000, 22000, 5000, 18000, 28000, 6500, 19000, 10000, 35000, 11000,
                 20000, 3500, 25000, 9000, 21000, 5500, 17000, 38000, 9500, 24000,
                 7000, 32000, 13000, 23000, 4000, 22500, 8500, 19500, 29000, 11500]
}
df=pd.DataFrame(data)

#explore the data
print("First 5 rows of data:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())

#visualize the data
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes=plt.subplots(1, 2, figsize=(14, 6))

scatter1=axes[0].scatter(df['pages'], df['rate_inr'],
                         c=df['rate_inr'], cmap='viridis', s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
axes[0].set_xlabel('Number of pages', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Rate (INR)', fontsize=12, fontweight='bold')
axes[0].set_title('Pages VS Rate', fontsize=14, fontweight='bold', pad=15)
plt.colorbar(scatter1, ax=axes[0], label='Rate Intensity')

scatter2 = axes[1].scatter(df['deadline_days'], df['rate_inr'],
                           c=df['rate_inr'], cmap='plasma', s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
axes[1].set_xlabel('Deadline (Days)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Rate (INR)', fontsize=12, fontweight='bold')
axes[1].set_title('Deadline vs Rate', fontsize=14, fontweight='bold', pad=15)
plt.colorbar(scatter2, ax=axes[1], label='Rate Intensity')

plt.tight_layout()
plt.show()

# preapre data for training
X=df[['pages','deadline_days']]
y=df['rate_inr']

# split into train and test sets
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=42)

# create and train the model
model=LinearRegression()
model.fit(X_train,y_train)

# check model parameters
print("\nModel Parameters:")
print(f"Coefficient: Pages={model.coef_[0]:.2f}, Deadline={model.coef_[1]:.2f}")
print(f"Intercept:{model.intercept_:.2f}")

# predictions on test data
y_pred=model.predict(X_test)

print("\nPredictions VS Actual:")
results=pd.DataFrame({'Actual':y_test.values, 'Predicted':y_pred.round()})
print(results)

# Model Accuracy
from sklearn.metrics import r2_score
score=r2_score(y_test, y_pred)
print(f"\nModel Accuracy (R2 Score):{score:.2f}")

# Answer priya's question
new_project=[[10, 5]]
predicted_rate=model.predict(new_project)
print(f"\nPriya's Question : 10-page website, 5-day deadline")
print(f"Recommended Rate: ₹{predicted_rate[0]:,.0f}")



