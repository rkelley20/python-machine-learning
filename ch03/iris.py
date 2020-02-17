from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# Load the iris dataset
iris = datasets.load_iris()

X = iris.data[:, [2, 3]]
y = iris.target

print(f'Class labels: {np.unique(y)}')

# Split into 30% test data and 70% training data
# stratify=y means that the function will return training and test subsets that
# have the same proportions of class labels as the input dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

print(f'Labels counts in y: {np.bincount(y)}')
print(f'Labels counts in y_train: {np.bincount(y_train)}')
print(f'Labels counts in y_test: {np.bincount(y_test)}')

sc = StandardScaler()
sc.fit(X_train) # Estimated the sample mean and standard deviation for each feature dimension
X_train_std = sc.transform(X_train) # Standardize training data using mu and sigma
X_test_std = sc.transform(X_test) # If we standardize training we must do so to test so the datasets are comparable

# eta0 is the learning rate
Perceptron()
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)

print(f'Misclassified samples: {(y_test != y_pred).sum()}')
print(f'Accuracy score: {accuracy_score(y_test, y_pred)}')

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan') 
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1 
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution)) 
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap) 
    plt.xlim(xx1.min(), xx1.max()) 
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)): 
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')
        # highlight test samples
        if test_idx:
            # plot all samples
            X_test, y_test = X[test_idx, :], y[test_idx]
    
    plt.scatter(X_test[:, 0], X_test[:, 1], c='', edgecolor='black', alpha=1.0, linewidth=1, marker='o', s=100, label='test set')

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined, ppn, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('patel width [standardized]')
plt.legend(loc='upper left')
plt.show()
