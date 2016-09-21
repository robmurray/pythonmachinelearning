import unittest

from skikitlearnclassifiers.DecisionTreeClassifier import DecisionTreeClassifier
from skikitlearnclassifiers.SupportVectorMachine import SupportVectorMachine
from skikitlearnclassifiers.DatasetLoader import DatasetLoader
from skikitlearnclassifiers.PerceptronClassifier import PerceptronClassifier
from skikitlearnclassifiers.LinearRegressionClassifier import LinearRegressionClassifier
from skikitlearnclassifiers.KernalSVMNonLinearProblem import KernalSVMNonLinearProblem
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
        ksvm.plot()

    def test_decisiontree(self):
        dtc = DecisionTreeClassifier(DatasetLoader(test_size=0.3, random_state=0))
        dtc.plot()
