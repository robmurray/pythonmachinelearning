import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from skikitlearnclassifiers.ClassifierBase import ClassifierBase
import logging

logger = logging.getLogger(__name__)


class LinearRegressionClassifier(ClassifierBase):

    def plot_sigmoid(self, save_image=False):
        z = np.arange(-7, 7, 0.1)
        phi_z = self.sigmoid(z)

        plt.plot(z, phi_z)
        plt.axvline(0.0, color='k')
        plt.ylim(-0.1, 1.1)
        plt.xlabel('z')
        plt.ylabel('$\phi (z)$')

        # y axis ticks and gridline
        plt.yticks([0.0, 0.5, 1.0])
        ax = plt.gca()
        ax.yaxis.grid(True)

        if save_image:
            plt.savefig('./figures/sigmoid.png', dpi=300)

        plt.show()

    def plot_cost(self, save_image=False):
        z = np.arange(-10, 10, 0.1)
        phi_z = self.sigmoid(z)

        c1 = [self.cost_1(x) for x in z]
        plt.plot(phi_z, c1, label='J(w) if y=1')

        c0 = [self.cost_0(x) for x in z]
        plt.plot(phi_z, c0, linestyle='--', label='J(w) if y=0')

        plt.ylim(0.0, 5.1)
        plt.xlim([0, 1])
        plt.xlabel('$\phi$(z)')
        plt.ylabel('J(w)')
        plt.legend(loc='best')
        if save_image:
            plt.savefig('./figures/log_cost.png', dpi=300)

        plt.show()

    def plot(self,save_image=False):
        logger.info('plotting linear regression classifier')
        lr = LogisticRegression(C=1000.0, random_state=0)
        lr.fit(self.X_train_std, self.y_train)
        logging.info('probability: %s',lr.predict_proba(self.X_test_std[0, :]))

        self.plot_decision_regions(self.X_combined_std, self.y_combined,classifier=lr, test_idx=range(105, 150))
        plt.xlabel('petal length [standardized]')
        plt.ylabel('petal width [standardized]')
        plt.legend(loc='upper left')
        plt.title('linear regression classifier')
        if save_image:
            plt.savefig('./figures/logistic_regression.png', dpi=300)
        plt.show()

    def plot_regression_path(self,save_image=False):
        logging.info('regularization')
        weights, params = [], []
        for c in np.arange(-5, 5):
            lr = LogisticRegression(C=10 ** c, random_state=0)
            lr.fit(self.X_train_std, self.y_train)
            weights.append(lr.coef_[1])
            params.append(10 ** c)

        weights = np.array(weights)
        plt.plot(params, weights[:, 0], label='petal length')
        plt.plot(params, weights[:, 1], linestyle='--', label='petal width')
        plt.ylabel('weight coefficient')
        plt.xlabel('C')
        plt.legend(loc='upper left')
        plt.xscale('log')
        if save_image:
            plt.savefig('./figures/regression_path.png', dpi=300)
        plt.show()


