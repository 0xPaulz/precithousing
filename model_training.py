import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


# load the data
df = pd.read_csv('Housing.csv')

# quick look
print(df.head())
print("\nshape:", df.shape)
print("\nmissing values:\n", df.isnull().sum())

# correlation heatmap (only numeric columns)
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('correlation heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# define which columns are categorical and which are numeric
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                    'airconditioning', 'prefarea', 'furnishingstatus']
numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

# preprocessing: scale numbers, one-hot encode categories
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ])

# features and target
X = df.drop('price', axis=1)
y = df['price']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# helper function to train and print results
def train_and_check(model, name):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"{name} → rmse: {rmse:,.0f} | r²: {r2:.3f}")
    return pipeline, rmse

# train three models
print("\ntraining models...\n")

linear_model, linear_rmse = train_and_check(LinearRegression(), "linear regression")
ridge_model,  ridge_rmse  = train_and_check(Ridge(alpha=1.0),      "ridge (l2)")
lasso_model,  lasso_rmse  = train_and_check(Lasso(alpha=100),     "lasso (l1)")

# pick the one with lowest rmse
best_model = min([
    ('linear', linear_model, linear_rmse),
    ('ridge',  ridge_model,  ridge_rmse),
    ('lasso',  lasso_model,  lasso_rmse)
], key=lambda x: x[2])

print(f"\nbest model: {best_model[0]} (rmse: {best_model[2]:,.0f})")

# save it for the streamlit app
joblib.dump(best_model[1], 'best_model.pkl')
print("model saved as best_model.pkl")

# quick plot of actual vs predicted for the best one
preds = best_model[1].predict(X_test)
plt.figure(figsize=(8,6))
plt.scatter(y_test, preds, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('actual price')
plt.ylabel('predicted price')
plt.title(f'actual vs predicted – {best_model[0]}')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.close()