from skikitlearnclassifiers.ClassifierUtils import ClassifierUtils as cu
import logging

'''
   

'''

logger = logging.getLogger(__name__)


class ClassifierBase(object):

    def __init__(self, datasetloader):
        self.classifierUtil= cu()
        self.datasetloader = datasetloader