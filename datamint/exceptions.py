class DatamintException(Exception):
    """
    Base class for exceptions in this module.
    """
    pass

class ResourceNotFoundError(DatamintException):
    """
    Exception raised when a resource is not found. 
    For instance, when trying to get a resource by a non-existing id.
    """

    def __init__(self,
                 resource_type: str,
                 params: dict):
        """ Constructor.

        Args:
            resource_type (str): A resource type.
            params (dict): Dict of params identifying the sought resource.
        """
        super().__init__()
        self.resource_type = resource_type
        self.params = params

    def set_params(self, resource_type: str, params: dict):
        self.resource_type = resource_type
        self.params = params

    def __str__(self):
        return f"Resource '{self.resource_type}' not found for parameters: {self.params}"

# Already existing (e.g, creating a project with a name that already exists)
class EntityAlreadyExistsError(DatamintException):
    """
    Exception raised when trying to create an entity that already exists.
    For instance, when creating a project with a name that already exists.
    """

    def __init__(self, entity_type: str, params: dict):
        """Constructor.

        Args:
            entity_type: The type of entity that already exists.
            params: Dict of params identifying the existing entity.
        """
        super().__init__()
        self.entity_type = entity_type
        self.params = params

    def __str__(self) -> str:
        return f"Entity '{self.entity_type}' already exists for parameters: {self.params}"