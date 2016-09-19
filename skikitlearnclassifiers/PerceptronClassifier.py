import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from skikitlearnclassifiers.ClassifierBase import ClassifierBase


class PerceptronClassifier(ClassifierBase):

    def plot(self):
        #Perceptron
        ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
        ppn.fit(self.X_train_std, self.y_train)
        print (ppn)
        print(self.y_test.shape)

        # results
        y_pred = ppn.predict(self.X_test_std)
        print('Misclassified samples: %d' % (self.y_test != y_pred).sum())
        print('Accuracy: %.2f' % accuracy_score(self.y_test, y_pred))

        #plot
        X_combined_std = np.vstack((self.X_train_std, self.X_test_std))
        y_combined = np.hstack((self.y_train, self.y_test))

        self.plot_decision_regions(X=X_combined_std, y=y_combined,classifier=ppn, test_idx=range(105, 150))

        plt.xlabel('petal length [standardized]')
        plt.ylabel('petal width [standardized]')
        plt.legend(loc='upper left')

        plt.tight_layout()
        # plt.savefig('./figures/iris_perceptron_scikit.png', dpi=300)
        plt.show()