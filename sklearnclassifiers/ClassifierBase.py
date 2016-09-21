from sklearnclassifiers.ClassifierUtils import ClassifierUtils as cu
import logging

'''
   Base class for doing a little exploratory programming with
   sklearn

'''

logger = logging.getLogger(__name__)


class ClassifierBase(object):

    def __init__(self, datasetloader):
        self.classifierUtil= cu()
        self.datasetloader = datasetloader