from .root_api_handler import RootAPIHandler
from .annotation_api_handler import AnnotationAPIHandler
from .apihandler.exp_api_handler import ExperimentAPIHandler


class APIHandler(RootAPIHandler, ExperimentAPIHandler, AnnotationAPIHandler):
    pass