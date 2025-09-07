import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# Choose model: "logistic", "svm", or "mlp"
model_type = "mlp"

# Step 1: Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Step 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train model
if model_type == "logistic":
    model = LogisticRegression(multi_class='ovr', max_iter=1000, random_state=42)
elif model_type == "svm":
    model = SVC(kernel='linear', random_state=42)
elif model_type == "mlp":
    model = MLPClassifier(hidden_layer_sizes=(50,30), max_iter=1000, random_state=42)
else:
    raise ValueError("Choose logistic, svm, or mlp")

model.fit(X_train_scaled, y_train)

# Step 5: Predict
y_pred = model.predict(X_test_scaled)

# Step 6: Evaluate
print(f"\nModel: {model_type.upper()}")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# Step 7: Visualize decision boundary (PCA 2D)
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train_scaled)
model.fit(X_train_2d, y_train)

x_min, x_max = X_train_2d[:,0].min()-1, X_train_2d[:,0].max()+1
y_min, y_max = X_train_2d[:,1].min()-1, X_train_2d[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_train_2d[:,0], X_train_2d[:,1], c=y_train, edgecolors='k')
plt.title(f'{model_type.upper()} Decision Boundary (PCA Iris)')
plt.show()
