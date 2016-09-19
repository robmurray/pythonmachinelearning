import unittest
from skikitlearnclassifiers import LinearRegressionClassifier
from skikitlearnclassifiers import PerceptronClassifier


class TestSkiKitLearnClassifiers(unittest.TestCase):

    def test_perceptron(self):
        pc = LinearRegressionClassifier.LinearRegressionClassifier()
        pc.plot()

    def test_linear_egression(self):
        lrc = PerceptronClassifier.PerceptronClassifier()
        lrc.plot()




