import matplotlib.pyplot as plt
import numpy as np
import logging
from sklearn.tree import DecisionTreeClassifier as dtc
from skikitlearnclassifiers.ClassifierBase import ClassifierBase

logger = logging.getLogger(__name__)


class DecisionTreeClassifier(ClassifierBase):

    def gini(self,p):
        return (p) * (1 - (p)) + (1 - p) * (1 - (1 - p))

    def entropy(self,p):
        return - p * np.log2(p) - (1 - p) * np.log2((1 - p))

    def error(self,p):
        return 1 - np.max([p, 1 - p])

    def plot(self):
        logger.info('using DecisionTree Classifier')

        x = np.arange(0.0, 1.0, 0.01)

        ent = [self.entropy(p) if p != 0 else None for p in x]
        sc_ent = [e * 0.5 if e else None for e in ent]
        err = [self.error(i) for i in x]

        fig = plt.figure()
        ax = plt.subplot(111)
        for i, lab, ls, c, in zip([ent, sc_ent, self.gini(x), err],
                                  ['Entropy', 'Entropy (scaled)',
                                   'Gini Impurity', 'Misclassification Error'],
                                  ['-', '-', '--', '-.'],
                                  ['black', 'lightgray', 'red', 'green', 'cyan']):
            line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),ncol=3, fancybox=True, shadow=False)

        ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
        ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
        plt.ylim([0, 1.1])
        plt.xlabel('p(i=1)')
        plt.ylabel('Impurity Index')
        # plt.savefig('./figures/impurity.png', dpi=300, bbox_inches='tight')
        plt.show()

        #fixme appear to be a library version conflict in Tree
        '''

        tree = dtc(criterion='entropy', max_depth=3, random_state=0)
        tree.fit(self.X_train, self.y_train)

        X_combined = np.vstack((self.X_train, self.X_test))
        y_combined = np.hstack((self.y_train, self.y_test))
        self.plot_decision_regions(X_combined, y_combined,classifier=tree, test_idx=range(105, 150))

        plt.xlabel('petal length [cm]')
        plt.ylabel('petal width [cm]')
        plt.legend(loc='upper left')
        # plt.savefig('./figures/decision_tree_decision.png', dpi=300)
        plt.show()

        '''