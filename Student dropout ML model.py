import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
df = pd.read_csv('xAPI-Edu-Data.csv')
np.shape(df)
df.info()
# Okay the Data is already clean
df['Class'].value_counts()
# I will eliminate l performance student so i store it in 1 and will not eliminate medium performance and high performance so it will be zero.
df['dropout'] = df['Class'].apply(lambda x: 1 if x == 'L' else 0)
X = df.drop(columns=['Class', 'dropout'],errors='ignore')
y=df['dropout']
categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(exclude='object').columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ]
)
model = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
risk_scores = model.predict_proba(X_test)[:, 1]
def risk_label(score):
    if score >= 0.7:
        return "High"
    elif score >= 0.4:
        return "Medium"
    else:
        return "Low"
results = pd.DataFrame({
    "student_id": X_test.index,
    "risk_score": risk_scores,
    "risk_label": [risk_label(risk_scores)],
    
    "predicted_dropout": (risk_scores >= 0.5).astype(int)
})

results.to_csv("predictions.csv", index=False)
joblib.dump(model, "dropout_model.pkl")

