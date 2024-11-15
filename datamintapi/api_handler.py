from .root_api_handler import RootAPIHandler
from .annotation_api_handler import AnnotationAPIHandler
from .experiment.exp_api_handler import ExperimentAPIHandler


class APIHandler(RootAPIHandler, ExperimentAPIHandler, AnnotationAPIHandler):
    pass