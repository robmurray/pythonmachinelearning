import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from skikitlearnclassifiers.ClassifierBase import ClassifierBase
import logging

logger = logging.getLogger(__name__)


class SupportVectorMachine(ClassifierBase):

    def plot(self):
        logger.info('plotting SVM')
        X_combined_std = np.vstack((self.X_train_std, self.X_test_std))
        y_combined = np.hstack((self.y_train, self.y_test))

        svm = SVC(kernel='linear', C=1.0, random_state=0)
        svm.fit(self.X_train_std, self.y_train)
        self.plot_decision_regions(X_combined_std, y_combined,classifier=svm, test_idx=range(105, 150))
        plt.xlabel('petal length [standardized]')
        plt.ylabel('petal width [standardized]')
        plt.legend(loc='upper left')
        plt.title('Maximum margin classification with support vector machines')
        #plt.tight_layout()
        # plt.savefig('./figures/support_vector_machine_linear.png', dpi=300)
        plt.show()

