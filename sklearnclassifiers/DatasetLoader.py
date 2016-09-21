import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import logging


logger = logging.getLogger(__name__)

'''
Loads, splits, standardizes the built-in sklearn iris dataset

'''


class DatasetLoader(object):

    def __init__(self, test_size=0.3, random_state=0):
        logger.info('initializing the dataset loader')
        self.test_size=test_size
        self.random_state=random_state
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
            train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)

        logger.info('standardizing features')
        sc = StandardScaler()
        sc.fit(self.X_train)
        self.X_train_std = sc.transform(self.X_train)
        self.X_test_std = sc.transform(self.X_test)
        self.X_combined_std = np.vstack((self.X_train_std, self.X_test_std))
        self.y_combined = np.hstack((self.y_train, self.y_test))