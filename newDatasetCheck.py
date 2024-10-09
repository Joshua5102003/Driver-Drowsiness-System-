import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pickle

data = pd.read_csv('ownDatasetCheck.csv')

X = data[['EAR', 'SIDE_TILT', 'FRONT_TILT']]
y = data['DROWSY']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, max_depth=15),
    "Support Vector Machine": SVC(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    with open(f'trained_{model_name}.pickle',"wb") as file:
        pickle.dump(model, file)
    print(f"{model_name} Accuracy: {accuracy}")
    print(f"Classification Report ({model_name}):\n{class_report}")
    print(f"Confusion Matrix ({model_name}):")
    print(conf_matrix)
