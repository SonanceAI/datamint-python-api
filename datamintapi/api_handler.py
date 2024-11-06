from .root_api_handler import RootAPIHandler
from .experiment.exp_api_handler import ExperimentAPIHandler


class APIHandler(RootAPIHandler, ExperimentAPIHandler):
    pass
