from .root_api_handler import RootAPIHandler
from .annotation_api_handler import AnnotationAPIHandler
from .exp_api_handler import ExperimentAPIHandler
from deprecated.sphinx import deprecated


@deprecated(reason="Please use `from datamint import Api` instead.", version="2.0.0")
class APIHandler(RootAPIHandler, ExperimentAPIHandler, AnnotationAPIHandler):
    """
    Deprecated. Use `from datamint import Api` instead.
    """
    pass