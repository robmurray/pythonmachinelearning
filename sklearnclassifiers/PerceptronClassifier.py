from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearnclassifiers.ClassifierBase import ClassifierBase
import logging

logger = logging.getLogger(__name__)


class PerceptronClassifier(ClassifierBase):

    def __init__(self, datasetloader):
        super(self.__class__, self).__init__(datasetloader)
        logger.debug('Perceptron')

    def predict(self,n_iter=40, eta0=0.1, random_state=0):
        ppn = Perceptron(n_iter=n_iter, eta0=eta0, random_state=random_state)
        ppn.fit(self.datasetloader.X_train_std, self.datasetloader.y_train)
        logger.debug('Perceptron: %s', ppn)

        y_pred = ppn.predict(self.datasetloader.X_test_std)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('results: %s', y_pred)
            logger.debug('Misclassified samples: %d' % (self.datasetloader.y_test != y_pred).sum())
            logger.debug('Accuracy: %.2f' % accuracy_score(self.datasetloader.y_test, y_pred))

        return y_pred

    def plot(self,n_iter=40, eta0=0.1, random_state=0,save_image=False):
        logger.info('plotting perseptron')
        ppn = Perceptron(n_iter=n_iter, eta0=eta0, random_state=random_state)
        ppn.fit(self.datasetloader.X_train_std, self.datasetloader.y_train)
        logger.debug('Perceptron: %s', ppn)

        self.classifierUtil.plot_decision_regions(self.datasetloader.X_combined_std,self.datasetloader.y_combined,classifier=ppn,test_idx=range(105, 150))

        plt.xlabel('petal length [standardized]')
        plt.ylabel('petal width [standardized]')
        plt.legend(loc='upper left')
        plt.title('Perceptron classifier')

        if save_image:
            plt.savefig('./figures/iris_perceptron_scikit.png', dpi=300)

        plt.show()