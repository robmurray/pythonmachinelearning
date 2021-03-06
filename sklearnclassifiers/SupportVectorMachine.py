import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearnclassifiers.ClassifierBase import ClassifierBase
import logging

logger = logging.getLogger(__name__)


class SupportVectorMachine(ClassifierBase):
    def __init__(self, datasetloader,save_image=False):
        super(self.__class__, self).__init__(datasetloader,save_image)
        logger.debug('SVM')

    def plot(self):
        logger.info('plotting SVM')

        svm = SVC(kernel='linear', C=1.0, random_state=0)
        svm.fit(self.datasetloader.X_train_std, self.datasetloader.y_train)
        self.classifierUtil.plot_decision_regions(self.datasetloader.X_combined_std, self.datasetloader.y_combined,classifier=svm, test_idx=range(105, 150))
        plt.xlabel('petal length [standardized]')
        plt.ylabel('petal width [standardized]')
        plt.legend(loc='upper left')
        plt.title('Maximum margin classification with support vector machines')
        if self.save_image:
            plt.savefig('./figures/support_vector_machine_linear.png', dpi=300)
        plt.show()

