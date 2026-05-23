import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, confusion_matrix, ConfusionMatrixDisplay)

#Load data
merged = pd.read_csv('merged.csv')
merged = merged.drop_duplicates(subset='SOC').reset_index(drop=True)
merged = merged.copy()  
print(f"Rows after deduplication: {len(merged)}")

#Define feature columns 
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

#Create binary target 
merged['HighRisk'] = (merged['Probability'] > 0.70).astype(int)
print(f"\nClass distribution:")
print(f"  Low risk  (0): {(merged['HighRisk'] == 0).sum()} jobs")
print(f"  High risk (1): {(merged['HighRisk'] == 1).sum()} jobs")

X = merged[feature_cols]
y = merged['HighRisk']

#Train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining rows: {X_train.shape[0]}")
print(f"Test rows:     {X_test.shape[0]}")

#Logistic Regression 
clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Metrics 
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)

print(f"\nLogistic Regression Results:")
print(f"  Accuracy:  {acc:.3f}")
print(f"  Precision: {prec:.3f}")
print(f"  Recall:    {rec:.3f}")

#Confusion matrix 
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Low Risk', 'High Risk'])

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Logistic Regression Confusion Matrix\n'
             f'Accuracy: {acc:.3f}  |  Precision: {prec:.3f}  |  Recall: {rec:.3f}')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()
print("Saved confusion_matrix.png")

#Top predictive features 
coefs = pd.Series(clf.coef_[0], index=feature_cols)
top10 = coefs.abs().sort_values(ascending=False).head(10)
top10_coefs = coefs[top10.index]

plt.figure(figsize=(9, 5))
colors = ['crimson' if c < 0 else 'steelblue' for c in top10_coefs]
plt.barh(top10_coefs.index[::-1], top10_coefs.values[::-1], color=colors[::-1])
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Top 10 Features — Logistic Regression Coefficients\n'
          '(Blue = predicts high risk, Red = predicts low risk)')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.savefig('classifier_features.png')
plt.close()
print("Saved classifier_features.png")