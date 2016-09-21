import unittest

from skikitlearnclassifiers import DecisionTreeClassifier
from skikitlearnclassifiers import LinearRegressionClassifier
from skikitlearnclassifiers import PerceptronClassifier
from skikitlearnclassifiers import SupportVectorMachine
from skikitlearnclassifiers import KernalSVMNonLinearProblem

'''
Many of these are technically not unit tests. However using the unit test framework is convenient way to
work with the various libraries
'''
class TestSkiKitLearnClassifiers(unittest.TestCase):

    def test_linear_regression(self):
        pc = LinearRegressionClassifier.LinearRegressionClassifier()
        pc.plot_sigmoid()
        pc.plot_cost()
        pc.plot()
        pc.plot_regression_path()

    def test_perceptron(self):
        lrc = PerceptronClassifier.PerceptronClassifier()
        y_pred = lrc.predict();
        lrc.plot()

    def test_svm(self):
        svm = SupportVectorMachine.SupportVectorMachine()
        svm.plot()

    def test_kernaksvm(self):
        ksvm = KernalSVMNonLinearProblem.KernalSVMNonLinearProblem()
        ksvm.plot()

    def test_decisiontree(self):
        dtc = DecisionTreeClassifier.DecisionTreeClassifier()
        dtc.plot()


