import pandas as pd
import joblib
import numpy as np 

from sklearn.model_selection import train_test_split    
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('Housing.csv')            #loading the dataset
df = pd.get_dummies(df, drop_first=True)   #converts all columns to numerical 

X = df.drop('price', axis=1)    #drops price column keaving all other columns as input
y = df['price']                 #what we ant to predict

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   #splitting data into training and testing sets


models = {
    "Linear Regression": LinearRegression(),
    "Ridge (L2)":        Ridge(alpha=1.0),      #defining the models in  dictionary
    "LASSO (L1)":        Lasso(alpha=1000)
}

best_name = None
best_model = None
best_rmse = float('inf')   

print("Training models...\n")

for name, model in models.items():  #iterating through the models
    model.fit(X_train, y_train)     #training the model
    pred = model.predict(X_test)    #making predictions for the test set
    
    rmse = np.sqrt(mean_squared_error(y_test, pred))    #calculating RMSE
    r2   = r2_score(y_test, pred)   #calculating R² score
    
    print(f"{name:18} → RMSE: {rmse:,.0f} | R²: {r2:.3f}")
    
    if rmse < best_rmse:           #checking for the best model
        best_rmse = rmse           #getting the name of the best model 
        best_model = model         #getting the best model itself     
        best_name = name

joblib.dump(best_model, 'best_model.pkl')   #saving the best model

print("\n" + "="*50)    #
print(f"BEST MODEL → {best_name}")
print(f"Test RMSE  → {best_rmse:,.0f}")
print(f"Test R²    → {r2_score(y_test, best_model.predict(X_test)):.3f}")
print("="*50)