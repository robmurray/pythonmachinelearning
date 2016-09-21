from sklearnclassifiers.ClassifierUtils import ClassifierUtils as cu
import logging

'''
   Base class for doing a little exploratory programming with
   sklearn

'''

logger = logging.getLogger(__name__)


class ClassifierBase(object):

    def __init__(self, datasetloader,save_image):
        self.classifierUtil= cu()
        self.datasetloader = datasetloader
        self.save_image=save_image