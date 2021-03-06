import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
import logging

from sklearnclassifiers.ClassifierBase import ClassifierBase

logger = logging.getLogger(__name__)


class KernalSVMNonLinearProblem(ClassifierBase):

    def __init__(self, datasetloader,save_image=False):
        super(self.__class__, self).__init__(datasetloader,save_image)
        logger.debug('Linear regression')

    def plot_scatter(self):
        logger.info('showing scatter plot')

        np.random.seed(0)
        X_xor = np.random.randn(200, 2)
        y_xor = np.logical_xor(X_xor[:, 0] > 0,
                               X_xor[:, 1] > 0)
        y_xor = np.where(y_xor, 1, -1)

        plt.scatter(X_xor[y_xor == 1, 0],
                    X_xor[y_xor == 1, 1],
                    c='b', marker='x',
                    label='1')
        plt.scatter(X_xor[y_xor == -1, 0],
                    X_xor[y_xor == -1, 1],
                    c='r',
                    marker='s',
                    label='-1')

        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        plt.legend(loc='best')
        if self.save_image:
            plt.savefig('./figures/xor.png', dpi=300)
        plt.show()

    def plot_xor(self,kernel='rbf', random_state=0, gamma=0.10, C=10.0):
        logger.info('using KernalSVM')
        np.random.seed(0)
        X_xor = np.random.randn(200, 2)
        y_xor = np.logical_xor(X_xor[:, 0] > 0,
                               X_xor[:, 1] > 0)
        y_xor = np.where(y_xor, 1, -1)

        svm = SVC(kernel=kernel, random_state=random_state, gamma=gamma, C=C)
        svm.fit(X_xor, y_xor)
        self.classifierUtil.plot_decision_regions(X_xor, y_xor,classifier=svm)

        plt.legend(loc='upper left')
        if self.save_image:
            plt.savefig('./figures/support_vector_machine_rbf_xor.png', dpi=300)
        plt.show()

    def plot(self,kernel='rbf', random_state=0, gamma=0.10, C=10.0):
        logger.info('using KernalSVM')
        svm = SVC(kernel=kernel, random_state=random_state, gamma=gamma, C=C)
        svm.fit(self.datasetloader.X_train_std, self.datasetloader.y_train)

        self.classifierUtil.plot_decision_regions(self.datasetloader.X_combined_std, self.datasetloader.y_combined,
                                                  classifier=svm, test_idx=range(105, 150))
        plt.xlabel('petal length [standardized]')
        plt.ylabel('petal width [standardized]')
        plt.legend(loc='upper left')
        if self.save_image:
            plt.savefig('./figures/support_vector_machine_rbf_iris_1.png', dpi=300)
        plt.show()
