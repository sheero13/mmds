import numpy as np, matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split
Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale
sc = StandardScaler()
Xtr, Xts = sc.fit_transform(Xtr), sc.transform(Xts)

# Train
model = LogisticRegression(multi_class='ovr').fit(Xtr, ytr)

# Predict
yp = model.predict(Xts)

# Accuracy + Report
print("Accuracy:", accuracy_score(yts, yp))
print(classification_report(yts, yp, target_names=iris.target_names))

# Plot (only first 2 features)
plt.scatter(Xts[:,0], Xts[:,1], c=yts, cmap='RdYlBu', edgecolor='k')
plt.xlabel('Sepal length'); plt.ylabel('Sepal width')
plt.title('Logistic Regression (Iris)')
plt.show()
