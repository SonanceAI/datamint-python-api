"""Run inference against a locally-running podman container. """
import json
import logging
from typing import TYPE_CHECKING, Any

import requests

from datamint.entities.annotations.annotation import Annotation
from datamint.entities.resource import LocalResource

if TYPE_CHECKING:
    from datamint.api.client import Api

logger = logging.getLogger(__name__)


def _prepare_local_resource(
    resource_id: str | None,
    file_path: str | None,
    api_client: 'Api | None',
) -> LocalResource:
    """Wrap a resource_id or a local file_path into a LocalResource, ready to serialize."""
    if (resource_id is None) == (file_path is None):
        raise ValueError("Exactly one of 'resource_id' or 'file_path' must be provided.")

    if file_path is not None:
        return LocalResource(local_filepath=file_path, convert_to_bytes=True)

    if api_client is None:
        raise ValueError("api_client is required when using resource_id.")
    
    resource = api_client.resources.get_by_id(resource_id)
    raw_data = resource.fetch_file_data(auto_convert=False)
    return LocalResource(raw_data=raw_data, **resource.model_dump(exclude_defaults=True))


def predict_local(
    container_port: int = 8080,
    *,
    resource_id: str | None = None,
    file_path: str | None = None,
    api_client: 'Api | None' = None,
    params: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> list[Annotation]:
    """Run inference against a podman container running on this machine.

    Args:
        container_port: Host port the container's ``/invocations`` endpoint
            is published on (e.g. via ``podman run -p 8080:8080 ...``).
        resource_id: ID of a resource already in Datamint. Mutually exclusive with file_path.
        file_path: Path to a local file to run inference on. Mutually exclusive with resource_id.
        api_client: Required only when resource_id is used, to fetch the
            resource's bytes.
        params: Extra parameters forwarded to the model (e.g. prediction
            mode).
        timeout: Seconds to wait for the prediction request.

    Returns:
        The predictions, parsed and validated as ``Annotation`` objects.
        Not saved to Datamint — this function only returns them.
    """
    resource = _prepare_local_resource(resource_id, file_path, api_client)

    payload: dict[str, Any] = {
        'inputs': [resource.model_dump(mode='json', exclude_defaults=True)],
    }
    if params:
        payload['params'] = params

    url = f"http://127.0.0.1:{container_port}/invocations"
    logger.info(f"Calling local model endpoint: {url}")
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=timeout,
    )
    response.raise_for_status()
    results = response.json()

    if 'error' in results:
        raise RuntimeError(f"Inference failed: {results['error']}")

    predictions = results.get('predictions', [])
    if not predictions:
        return []

    annotations: list[Annotation] = []
    for annot_json in predictions[0]:
        annot = Annotation.model_validate(annot_json)
        if annot.resource_id and resource.id and annot.resource_id != resource.id:
            raise ValueError(
                f"Annotation resource_id {annot.resource_id!r} does not match "
                f"the input resource id {resource.id!r}."
            )
        if not annot.resource_id:
            annot.resource_id = resource.id
        annotations.append(annot)

    return annotations
