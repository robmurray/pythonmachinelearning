import unittest

from sklearnclassifiers.DecisionTreeClassifier import DecisionTreeClassifier
from sklearnclassifiers.SupportVectorMachine import SupportVectorMachine
from sklearnclassifiers.DatasetLoader import DatasetLoader
from sklearnclassifiers.PerceptronClassifier import PerceptronClassifier
from sklearnclassifiers.LinearRegressionClassifier import LinearRegressionClassifier
from sklearnclassifiers.KernalSVMNonLinearProblem import KernalSVMNonLinearProblem

'''
Many of these are technically not unit tests. However using the unit test framework is convenient way to
work with the various libraries
'''


class TestSkiKitLearnClassifiers(unittest.TestCase):

    def test_linear_regression(self):
        pc = LinearRegressionClassifier(DatasetLoader(test_size=0.3, random_state=0))
        pc.plot_sigmoid()
        pc.plot_cost()
        pc.plot()
        pc.plot_regression_path()

    def test_perceptronClassifier(self):
        pc= PerceptronClassifier(DatasetLoader(test_size=0.3, random_state=0))
        pc.predict()
        pc.plot()

    def test_svm(self):
        svm = SupportVectorMachine(DatasetLoader(test_size=0.3, random_state=0))
        svm.plot()

    def test_kernaksvm(self):
        ksvm = KernalSVMNonLinearProblem(DatasetLoader(test_size=0.3, random_state=0))
        ksvm.plot_scatter()
        ksvm.plot_xor(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
        ksvm.plot(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
        ksvm.plot(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
        ksvm.plot(kernel='rbf', random_state=0, gamma=100.0, C=1.0)

    def test_decisiontree(self):
        dtc = DecisionTreeClassifier(DatasetLoader(test_size=0.3, random_state=0))
        dtc.plot()
