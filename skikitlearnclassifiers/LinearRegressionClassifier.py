import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from skikitlearnclassifiers.ClassifierBase import ClassifierBase


class LinearRegressionClassifier(ClassifierBase):

    def plot(self):
        X_combined_std = np.vstack((self.X_train_std, self.X_test_std))
        y_combined = np.hstack((self.y_train, self.y_test))

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

        plt.tight_layout()
        # plt.savefig('./figures/sigmoid.png', dpi=300)
        plt.show()

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
        plt.tight_layout()
        # plt.savefig('./figures/log_cost.png', dpi=300)
        plt.show()

        lr = LogisticRegression(C=1000.0, random_state=0)
        lr.fit(self.X_train_std, self.y_train)

        self.plot_decision_regions(X_combined_std, y_combined,classifier=lr, test_idx=range(105, 150))
        plt.xlabel('petal length [standardized]')
        plt.ylabel('petal width [standardized]')
        plt.legend(loc='upper left')
        plt.tight_layout()
        # plt.savefig('./figures/logistic_regression.png', dpi=300)
        plt.show()
