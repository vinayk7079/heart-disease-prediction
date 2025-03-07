import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load the dataset
data = pd.read_csv('data.csv')

# Preprocessing
data.dropna(inplace=True)  # Drop missing values
X = data.drop('target', axis=1)  # Features
y = data['target']  # Target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define individual models
logistic = LogisticRegression(max_iter=1000, random_state=42)
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
decision_tree = DecisionTreeClassifier(random_state=42)
svm = SVC(probability=True, random_state=42)
knn = KNeighborsClassifier()

# Create ensemble model with soft voting
ensemble_model = VotingClassifier(
    estimators=[
        ('lr', logistic),
        ('rf', random_forest),
        ('dt', decision_tree),
        ('svm', svm),
        ('knn', knn)
    ],
    voting='soft'
)

# Train the model
ensemble_model.fit(X_train_scaled, y_train)

# Evaluate the model
accuracy = ensemble_model.score(X_test_scaled, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model and scaler
joblib.dump(ensemble_model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("Model and scaler saved successfully.")