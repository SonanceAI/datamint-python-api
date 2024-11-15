from .root_api_handler import RootAPIHandler
from .annotation_api_handler import AnnotationAPIHandler


class APIHandler(RootAPIHandler, AnnotationAPIHandler):
    pass