# import
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# load data
df = pd.read_csv('train_and_test2.csv')
X = df[['Age', 'Fare', 'Sex', 'sibsp', 'Pclass']]
y = df['2urvived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 42
)

model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"accuracy: {accuracy * 100:.2f}%")
print("Logs:")
print(classification_report(y_test, predictions, target_names=['Dead (0)', 'Survived (1)']))
