import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#Load merged data
merged = pd.read_csv('merged.csv')

#Define columns 
state_cols = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado',
              'Connecticut','Delaware','District of Columbia','Florida','Georgia',
              'Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky',
              'Louisiana','Maine','Maryland','Massachusetts','Michigan','Minnesota',
              'Mississippi','Missouri','Montana','Nebraska','Nevada','New Hampshire',
              'New Jersey','New Mexico','New York','North Carolina','North Dakota',
              'Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island',
              'South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont',
              'Virginia','Washington','West Virginia','Wisconsin','Wyoming']

id_cols = ['SOC', 'Occupation', 'O*NET-SOC Code']
drop_cols = state_cols + id_cols

feature_cols = [c for c in merged.columns if c not in drop_cols and c != 'Probability']

X = merged[feature_cols]
y = merged['Probability']

#Train/test split (80/20) 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training rows: {X_train.shape[0]}")
print(f"Test rows:     {X_test.shape[0]}")

# Ridge Regression 
ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

r2_ridge = r2_score(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))

print(f"\nRidge Results:")
print(f"  Best alpha: {ridge.alpha_}")
print(f"  R²:         {r2_ridge:.3f}")
print(f"  RMSE:       {rmse_ridge:.3f}")

# Lasso Regression 
lasso = LassoCV(alphas=[0.001, 0.01, 0.1, 1], cv=5, max_iter=10000)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

r2_lasso = r2_score(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))

print(f"\nLasso Results:")
print(f"  Best alpha: {lasso.alpha_:.4f}")
print(f"  R²:         {r2_lasso:.3f}")
print(f"  RMSE:       {rmse_lasso:.3f}")

# Feature importance (top 15 from Lasso) 
lasso_coefs = pd.Series(lasso.coef_, index=feature_cols)
top15 = lasso_coefs.abs().sort_values(ascending=False).head(15)
top15_coefs = lasso_coefs[top15.index]

plt.figure(figsize=(10, 6))
colors = ['crimson' if c < 0 else 'steelblue' for c in top15_coefs]
plt.barh(top15_coefs.index[::-1], top15_coefs.values[::-1], color=colors[::-1])
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Top 15 Features by Lasso Coefficient\n(Blue = protective, Red = increases risk)')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()
print("\nSaved feature_importance.png")

#Predicted vs Actual plot (Ridge) 
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred_ridge, alpha=0.5, color='steelblue', edgecolors='white')
plt.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Perfect prediction')
plt.xlabel('Actual Automation Probability')
plt.ylabel('Predicted Automation Probability')
plt.title(f'Ridge: Predicted vs Actual  (R² = {r2_ridge:.3f})')
plt.legend()
plt.tight_layout()
plt.savefig('predicted_vs_actual.png')
plt.close()
print("Saved predicted_vs_actual.png")