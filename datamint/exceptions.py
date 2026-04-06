class DatamintException(Exception):
    """
    Base class for exceptions in this module.
    """
    pass


class ItemNotFoundError(DatamintException):
    """
    Exception raised when an item is not found. 
    For instance, when trying to get an item by a non-existing id.
    """

    def __init__(self,
                 item_type: str,
                 params: dict):
        """ Constructor.

        Args:
            item_type (str): An item type.
            params (dict): Dict of params identifying the sought item.
        """
        self.item_type = item_type
        self.params = params

    @property
    def resource_type(self):
        return self.item_type

    @resource_type.setter
    def resource_type(self, value: str):  # Alias for backward compatibility. To be removed in a future major version.
        self.item_type = value

    def set_params(self, resource_type: str, params: dict):
        self.item_type = resource_type
        self.params = params

    def __str__(self):
        return f"Item '{self.item_type}' not found for parameters: {self.params}"


ResourceNotFoundError = ItemNotFoundError  # Alias for backward compatibility. To be removed in a future major version.


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
