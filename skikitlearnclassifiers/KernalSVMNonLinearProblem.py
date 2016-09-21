import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
import logging

from skikitlearnclassifiers.ClassifierBase import ClassifierBase

logger = logging.getLogger(__name__)


class KernalSVMNonLinearProblem(ClassifierBase):

    def plot(self):
        logger.info('using KernalSVM')

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
        # plt.savefig('./figures/xor.png', dpi=300)
        plt.show()

        svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
        svm.fit(X_xor, y_xor)
        self.plot_decision_regions(X_xor, y_xor,classifier=svm)

        plt.legend(loc='upper left')
        # plt.savefig('./figures/support_vector_machine_rbf_xor.png', dpi=300)
        plt.show()

        svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
        svm.fit(self.X_train_std, self.y_train)

        self.plot_decision_regions(self.X_combined_std, self.y_combined,classifier=svm, test_idx=range(105, 150))
        plt.xlabel('petal length [standardized]')
        plt.ylabel('petal width [standardized]')
        plt.legend(loc='upper left')
        # plt.savefig('./figures/support_vector_machine_rbf_iris_1.png', dpi=300)
        plt.show()

        svm = SVC(kernel='rbf', random_state=0, gamma=100.0, C=1.0)
        svm.fit(self.X_train_std, self.y_train)

        self.plot_decision_regions(self.X_combined_std, self.y_combined,classifier=svm, test_idx=range(105, 150))
        plt.xlabel('petal length [standardized]')
        plt.ylabel('petal width [standardized]')
        plt.legend(loc='upper left')
        # plt.savefig('./figures/support_vector_machine_rbf_iris_2.png', dpi=300)
        plt.show()