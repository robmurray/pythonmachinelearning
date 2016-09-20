import unittest

from skikitlearnclassifiers import DecisionTreeClassifier
from skikitlearnclassifiers import LinearRegressionClassifier
from skikitlearnclassifiers import PerceptronClassifier
from skikitlearnclassifiers import SupportVectorMachine
from skikitlearnclassifiers import KernalSVMNonLinearProblem


class TestSkiKitLearnClassifiers(unittest.TestCase):

    def test_linear_regression(self):
        pc = LinearRegressionClassifier.LinearRegressionClassifier()
        pc.plot()

    def test_perceptron(self):
        lrc = PerceptronClassifier.PerceptronClassifier()
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


