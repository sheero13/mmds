import cv2, os, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Dataset path and categories
data = './Car-Bike-Dataset'
cats = ['Bike','Car']

X, y = [], []
for i,c in enumerate(cats):
    for f in os.listdir(os.path.join(data,c)):
        img = cv2.imread(os.path.join(data,c,f),0)  # read gray
        if img is None: continue
        X.append(cv2.resize(img,(64,64)).flatten())
        y.append(i)

X, y = np.array(X), np.array(y)
xtr, xts, ytr, yts = train_test_split(X,y,test_size=0.2)
clf = SVC(kernel='linear').fit(xtr,ytr)
yp = clf.predict(xts)

print("Accuracy:",accuracy_score(yts,yp))

# Show some predictions
for i in range(9):
    k=np.random.randint(0,len(xts))
    plt.subplot(3,3,i+1)
    plt.imshow(xts[k].reshape(64,64),cmap='gray')
    plt.title(f"T:{cats[yts[k]]}/P:{cats[yp[k]]}")
    plt.axis('off')
plt.show()
