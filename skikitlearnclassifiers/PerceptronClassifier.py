from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from skikitlearnclassifiers.ClassifierBase import ClassifierBase
import logging

logger = logging.getLogger(__name__)


class PerceptronClassifier(ClassifierBase):

    def __init__(self, n_iter=40, eta0=0.1, random_state=0):
        super(PerceptronClassifier, self,).__init__()
        self.ppn = Perceptron(n_iter=n_iter, eta0=eta0, random_state=random_state)
        self.ppn.fit(self.X_train_std, self.y_train)
        logger.debug('Perceptron: %s', self.ppn)

    def predict(self):
        y_pred = self.ppn.predict(self.X_test_std)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('results: %s', y_pred)
            logger.debug('Misclassified samples: %d' % (self.y_test != y_pred).sum())
            logger.debug('Accuracy: %.2f' % accuracy_score(self.y_test, y_pred))

        return y_pred

    def plot(self,save_image=False):
        logger.info('plotting perseptron')

        self.plot_decision_regions(X=self.X_combined_std, y=self.y_combined,classifier=self.ppn, test_idx=range(105, 150))

        plt.xlabel('petal length [standardized]')
        plt.ylabel('petal width [standardized]')
        plt.legend(loc='upper left')
        plt.title('Perceptron classifier')

        if save_image:
            plt.savefig('./figures/iris_perceptron_scikit.png', dpi=300)

        plt.show()