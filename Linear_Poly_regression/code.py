#importing the libraries!
import numpy as np;
import matplotlib.pyplot as plt;
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

#generate Synthetic data!
X=np.linspace(0,10,100).reshape(-1,1)
y_true= X**3

#adding random noise!
noise=np.random.normal(0, 50 , X.shape)
y=y_true+noise

#splitting the data!
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

model=LinearRegression()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
mse=mean_squared_error(y_test, y_pred)

train_mses=[]
test_mses=[]
degrees=range(1,6)

for degree in degrees:
    poly=PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly=poly.fit_transform(X_train)
    X_test_poly=poly.transform(X_test)

    model=LinearRegression()
    model.fit(X_train_poly,y_train)

    y_train_pred=model.predict(X_train_poly)
    y_test_pred=model.predict(X_test_poly)

    mse_train=mean_squared_error(y_train, y_train_pred)
    mse_test=mean_squared_error(y_test, y_test_pred)

    train_mses.append(mse_train)
    test_mses.append(mse_test)
    print(f"Degree: {degree} | Train MSE={mse_train:.3f} | Test MSE={mse_test:.3f}")

#plot the training vs testing error!
plt.figure(figsize=(10,6))
plt.plot(degrees, train_mses, label='train_mses', marker='o', color='blue')
plt.plot(degrees, test_mses, label='test_mses', marker='o', color='red')
plt.xlabel('degrees')
plt.ylabel('MSE')
plt.title('Training VS Testing Error')
plt.legend()
plt.savefig('Train_vs_Test_Error.png')
plt.show()


#plot the datapoints!
X_plot=np.linspace(0,10,100).reshape(-1,1)
plt.figure(figsize=(10,6))
plt.scatter(X, y, color='gray', alpha=0.5, label='Data Points')
for degree in [1, 3, 5]:
    poly=PolynomialFeatures(degree=degree, include_bias=False)
    X_plot_poly=poly.fit_transform(X_plot)
    X_train_poly=poly.fit_transform(X_train)

    model=LinearRegression()
    model.fit(X_train_poly, y_train)

    y_plot_pred=model.predict(X_plot_poly)

    plt.plot(X_plot, y_plot_pred, label=f'Degree:{degree}')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression Fits')
plt.legend()
plt.savefig('Poly_reg_fits.png')
plt.show()
   

