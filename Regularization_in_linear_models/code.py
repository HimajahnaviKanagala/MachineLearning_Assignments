#importing libraries!
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

#loading the dataset!
df=pd.read_csv("House Price Prediction Dataset.csv")

#Displaying the first few rows!
print(df.head())

#checking data types
print(df.dtypes)

#checking for missing values
print(df.isnull().sum())

#selecting numerical features!
X=df[["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt"]]

#selecting target variable!
y=df[["Price"]]

print(X.shape)
print(y.shape)

#train-test-split!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled.mean(axis=0))
print(X_train_scaled.std(axis=0))

#implementing basic model-- Linear model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

y_train_pred=linear_model.predict(X_train_scaled)
y_test_pred=linear_model.predict(X_test_scaled)

train_mse=mean_squared_error(y_train, y_train_pred)
test_mse=mean_squared_error(y_test, y_test_pred)

print("Training MSE:", train_mse)
print("Testing MSE:", test_mse)
print(linear_model.coef_)

#implementing ridge(L2 Regularization)!
alphas=np.logspace(-3, 4, 100)
ridge_train_mse=[]
ridge_test_mse=[]
ridge_coefficients=[]

for alpha in alphas:
    ridge= Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)

    y_train_pred=ridge.predict(X_train_scaled)
    y_test_pred=ridge.predict(X_test_scaled)

    ridge_train_mse.append(mean_squared_error(y_train, y_train_pred))
    ridge_test_mse.append(mean_squared_error(y_test, y_test_pred))
    ridge_coefficients.append(ridge.coef_)

#plot the training vs testing error in ridge
plt.figure(figsize=(10,6))
plt.plot(alphas, ridge_train_mse, label='Training MSE')
plt.plot(alphas, ridge_test_mse, label='Testing MSE')
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("MSE")
plt.title("Ridge Regression: Training VS Testing Error")
plt.legend()
plt.savefig("ridge_train_test_error.png", dpi=300, bbox_inches="tight")
plt.show()

#ridge coefficient path
feature_names=X.columns
plt.figure(figsize=(10,6))
for coef, feature in zip(np.array(ridge_coefficients).T, feature_names):
    plt.plot(alphas, coef, label=feature)
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("Coefficient Value")
plt.title("Ridge Coefficient Shrinkage Path")
plt.legend()
plt.savefig("ridge_coeffcient_path.png", dpi=300, bbox_inches="tight")
plt.show()

#implementing lasso(L1 Regularization!)
lasso_train_mse=[]
lasso_test_mse=[]
lasso_coefficients=[]

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)

    y_train_pred=lasso.predict(X_train_scaled)
    y_test_pred=lasso.predict(X_test_scaled)

    lasso_train_mse.append(mean_squared_error(y_train, y_train_pred))
    lasso_test_mse.append(mean_squared_error(y_test, y_test_pred))
    lasso_coefficients.append(lasso.coef_)

#plot the training vs testing error in lasso
plt.figure(figsize=(10,6))
plt.plot(alphas, lasso_train_mse, label='Training MSE')
plt.plot(alphas, lasso_test_mse, label='Testing MSE')
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("MSE")
plt.title("Lasso Regression: Training VS Testing Error")
plt.legend()
plt.savefig("lasso_train_test_error.png", dpi=300, bbox_inches="tight")
plt.show()

#lasso coefficient path
feature_names=X.columns
plt.figure(figsize=(10,6))
for coef, feature in zip(np.array(lasso_coefficients).T, feature_names):
    plt.plot(alphas, coef, label=feature)
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("Coefficient Value")
plt.title("Lasso Coefficient Path")
plt.legend()
plt.savefig("lasso_coeffcient_path.png", dpi=300, bbox_inches="tight")
plt.show()

#implementing elastic net
l1_ratio= 0.5
enet_train_mse=[]
enet_test_mse=[]
enet_coefficients=[]

for alpha in alphas:
    enet= ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    enet.fit(X_train_scaled, y_train)

    y_train_pred=enet.predict(X_train_scaled)
    y_test_pred=enet.predict(X_test_scaled)

    enet_train_mse.append(mean_squared_error(y_train, y_train_pred))
    enet_test_mse.append(mean_squared_error(y_test, y_test_pred))
    enet_coefficients.append(enet.coef_)

#plot the training vs testing error in elasticnet
plt.figure(figsize=(10,6))
plt.plot(alphas, enet_train_mse, label='Training MSE')
plt.plot(alphas, enet_test_mse, label='Testing MSE')
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("MSE")
plt.title("enet Regression: Training VS Testing Error")
plt.legend()
plt.savefig("elasticnet_train_test_error.png", dpi=300, bbox_inches="tight")
plt.show()

#elastic net coefficient path
feature_names=X.columns
plt.figure(figsize=(10,6))
for coef, feature in zip(np.array(enet_coefficients).T, feature_names):
    plt.plot(alphas, coef, label=feature)
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("Coefficients Value")
plt.title("Elastic Net Coefficient Path")
plt.legend()
plt.savefig("elasticnet_coeffcient_path.png", dpi=300, bbox_inches="tight")
plt.show()


