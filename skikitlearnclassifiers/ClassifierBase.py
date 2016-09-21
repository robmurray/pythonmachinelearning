from skikitlearnclassifiers.ClassifierUtils import ClassifierUtils as cu
import logging

'''
    A base class for exploring skikit-learn classifiers.
      - loads iris dataset
      - split data into training and test data
      - performs feature standardization
      - defines useful methods for predictions and plotting of data

'''

logger = logging.getLogger(__name__)


class ClassifierBase(object):

    def __init__(self, datasetloader):
        self.classifierUtil= cu()
        self.datasetloader = datasetloader