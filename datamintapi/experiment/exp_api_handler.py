from datamintapi import APIHandler

class ExperimentAPIHandler(APIHandler):
    def __init__(self, root_url: str, api_key: str | None = None):
        super().__init__(root_url, api_key)

