from .root_api_handler import RootAPIHandler
from .annotation_api_handler import AnnotationAPIHandler
from .exp_api_handler import ExperimentAPIHandler


class APIHandler(RootAPIHandler, ExperimentAPIHandler, AnnotationAPIHandler):
    """
    Import using this code:

    .. code-block:: python
    
        from datamint import APIHandler
        api = APIHandler()
    """
    pass