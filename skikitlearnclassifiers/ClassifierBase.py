import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings
import logging

'''
    A base class for exploring skikit-learn classifiers.
      - loads iris dataset
      - split data into training and test data
      - performs feature standardization
      - defines useful methods for predictions and plotting of data

'''

logger = logging.getLogger(__name__)


class ClassifierBase(object):

    def __init__(self):
        logger.info('loading iris dataset')

        self.iris = datasets.load_iris()
        self.X = self.iris.data[:, [2, 3]]
        self.y = self.iris.target

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('X shape: %s',self.X.shape)
            logger.debug('Y shape: %s', self.y.shape)
            logger.debug('Class labels: %s', np.unique(self.y))

        # split into training and test data
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.3, random_state=0)

        logger.info('standardizing features')
        sc = StandardScaler()
        sc.fit(self.X_train)
        self.X_train_std = sc.transform(self.X_train)
        self.X_test_std = sc.transform(self.X_test)
        self.X_combined_std = np.vstack((self.X_train_std, self.X_test_std))
        self.y_combined = np.hstack((self.y_train, self.y_test))

    def cost_1(self,z):
        cost = - np.log(self.sigmoid(z))
        logger.debug('cost1: %s', cost)
        return cost

    def cost_0(self,z):
        cost2 = - np.log(1 - self.sigmoid(z))
        logger.debug('cost2: %s', cost2)
        return cost2

    def sigmoid(self,z):
        sigmoid=1.0 / (1.0 + np.exp(-z))
        logger.debug('sigmoid: %s', sigmoid)
        return sigmoid

    def versiontuple(self,v):
        rettuple =tuple(map(int, (v.split("."))))
        if logger.isEnabledFor(logging.DEBUG):
            logging.debug('version: %s',v)
            logging.debug('tuple: %s', rettuple)

        return rettuple

    def plot_decision_regions(self,X, y, classifier, test_idx=None, resolution=0.02):

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
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                        alpha=0.8, c=cmap(idx),
                        marker=markers[idx], label=cl)

        # highlight test samples
        if test_idx:
            # plot all samples
            if not self.versiontuple(np.__version__) >= self.versiontuple('1.9.0'):
                self.X_test, self.y_test = X[list(test_idx), :], y[list(test_idx)]
                warnings.warn('Please update to NumPy 1.9.0 or newer')
            else:
                self.X_test, self.y_test = X[test_idx, :], y[test_idx]

            plt.scatter(self.X_test[:, 0],
                        self.X_test[:, 1],
                        c='',
                        alpha=1.0,
                        linewidths=1,
                        marker='o',
                        s=55, label='test set')
