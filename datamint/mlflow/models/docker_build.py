""" Build a podman image for an MLflow model locally """
import logging
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path

import mlflow.artifacts
import requests
from mlflow.models.model import Model
from mlflow.pyfunc.backend import _MODEL_DIR_NAME
from mlflow.utils.environment import _REQUIREMENTS_FILE_NAME, _get_requirements_from_file
from mlflow.version import VERSION as MLFLOW_VERSION

from datamint.configs import DEFAULT_DEPLOY_MODEL_ALIAS

logger = logging.getLogger(__name__)

_PODMAN_NOT_RUNNING_MSG = (
    "Could not connect to podman. Make sure podman is installed and its service is running "
)

_PODMAN_CLI_NOT_FOUND_MSG = (
    "The 'podman' command was not found on PATH. Install podman "
    "(https://podman.io/docs/installation) before building images locally."
)

_LINKED_MODELS_DIRNAME = "linked_models"

_PYTORCH_BASE_IMAGES = {
    '2.10.0': '2.10.0-cuda13.0-cudnn9-runtime',
    '2.9.1': '2.9.1-cuda13.0-cudnn9-runtime',
    '2.9.0': '2.9.0-cuda12.8-cudnn9-runtime',
    '2.8.0': '2.8.0-cuda12.8-cudnn9-runtime',
    '2.7.1': '2.7.1-cuda12.8-cudnn9-runtime',
    '2.7.0': '2.7.0-cuda12.8-cudnn9-runtime',
    '2.6.0': '2.6.0-cuda11.8-cudnn9-runtime',
    '2.5.1': '2.5.1-cuda11.8-cudnn9-runtime',
    '2.5.0': '2.5.0-cuda11.8-cudnn9-runtime',
    '2.4.1': '2.4.1-cuda11.8-cudnn9-runtime',
}
_PYTORCH_BASE_IMAGES = {k: f"pytorch/pytorch:{v}" for k, v in _PYTORCH_BASE_IMAGES.items()}

_DEFAULT_PYTHON_IMAGE = 'python:3.12-slim'


def _download_model(model_uri: str, dst_path: str) -> str:
    """Download an MLflow model's artifacts to a local directory.

    Args:
        model_uri: MLflow model URI, e.g. ``models:/MyModel@champion``.
        dst_path: Local directory to download the model files into.

    Returns:
        The local path the model was downloaded to (same as ``dst_path``).
    """
    return mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=dst_path)


def _load_model_requirements(model_dir: Path) -> list:
    """Read the model's ``requirements.txt`` (if present) into parsed ``Requirement`` objects."""
    requirements_path = model_dir / _REQUIREMENTS_FILE_NAME
    if not requirements_path.exists():
        return []
    return _get_requirements_from_file(requirements_path)


def _select_base_image(requirements: list, with_gpu: bool) -> str:
    """Pick a base container image: plain Python, or a matching PyTorch/CUDA image. """
    if not with_gpu:
        return _DEFAULT_PYTHON_IMAGE

    for req in requirements:
        if req.name == 'torch' and '+cpu' not in str(req):
            valid_versions = [v for v in _PYTORCH_BASE_IMAGES.keys() if v in req.specifier]
            if len(valid_versions) == 0:
                logger.warning(f"No valid PyTorch version found in requirements: {req.specifier}.")
                return _DEFAULT_PYTHON_IMAGE
            
            logger.info(f"Found valid PyTorch versions: {valid_versions}")
            pytorch_version = sorted(valid_versions, key=lambda v: tuple(map(int, v.split('.'))))[-1]
            logger.info(f"Using PyTorch version: {pytorch_version}")
            return _PYTORCH_BASE_IMAGES[pytorch_version]
    else:
        logger.info("No PyTorch dependency found in requirements. Using default Python image.")
        return _DEFAULT_PYTHON_IMAGE


def _write_dockerignore(output_dir: str) -> None:
    """Write a .dockerignore file to minimize the podman build context."""
    dockerignore_content = "\n".join([
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        "**/.git",
        "**/.gitignore",
        "**/.DS_Store",
        "**/Thumbs.db",
        "**/*.egg-info",
        "",
    ])
    (Path(output_dir) / ".dockerignore").write_text(dockerignore_content)


def _read_package_resource(filename: str) -> str:
    """Read a text file bundled alongside this module (e.g. ``serve.py``, ``Dockerfile.template``)."""
    resource = files(__package__) / filename
    with as_file(resource) as path:
        return path.read_text()


def _generate_dockerfile(model_uri: str, output_dir: str, with_gpu: bool) -> Model:
    """Download the model and write a ``Dockerfile`` (+ ``serve.py``) into output_dir.

    Args:
        model_uri: MLflow model URI, e.g. ``models:/MyModel@champion``.
        output_dir: Empty directory to assemble the podman build context in.
        with_gpu: Whether to build a GPU (CUDA) capable image.

    Returns:
        The loaded ``Model`` metadata.
    """
    model_dir = Path(output_dir) / _MODEL_DIR_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = Path(_download_model(model_uri, str(model_dir)))

    mlflow_model = Model.load(downloaded_path)
    requirements = _load_model_requirements(downloaded_path)
    base_image = _select_base_image(requirements, with_gpu=with_gpu)
    logger.info(f"Using base image: {base_image}")

    (Path(output_dir) / _LINKED_MODELS_DIRNAME).mkdir(parents=True, exist_ok=True)
    _write_dockerignore(output_dir)

    serve_py_content = _read_package_resource('serve.py')
    (Path(output_dir) / 'serve.py').write_text(serve_py_content)

    dockerfile_template = _read_package_resource('Dockerfile.template')

    if base_image.startswith("python:"):
        setup_python_venv_steps = (
            "RUN apt-get -y update && apt-get install -y --no-install-recommends nginx "
            "libxcb1 libgl1 libglib2.0-0 "
            " && apt-get clean && rm -rf /var/lib/apt/lists/*\n"
        )
    elif base_image.startswith("pytorch/pytorch:"):
        setup_python_venv_steps = (
            "RUN apt-get -y update "
            "&& DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y --no-install-recommends "
            "wget nginx bzip2 libxcb1 libgl1 libglib2.0-0 "
            "&& apt-get clean && rm -rf /var/lib/apt/lists/*\n\n"
        )
    else:
        logger.warning(
            f"Base image '{base_image}' is not recognized as a standard Python or PyTorch image. "
            "Skipping additional setup steps."
        )
        setup_python_venv_steps = ""

    if with_gpu:
        nvidia_env_vars = (
            "# NVIDIA Container Runtime environment variables\n"
            "ENV NVIDIA_VISIBLE_DEVICES=all \\\n"
            "    NVIDIA_DRIVER_CAPABILITIES=compute,utility \\\n"
            "    MLFLOW_DEFAULT_PREDICTION_DEVICE=cuda\n"
        )
    else:
        nvidia_env_vars = "ENV MLFLOW_DEFAULT_PREDICTION_DEVICE=cpu\n"

    dockerfile = dockerfile_template.format(
        base_image=base_image,
        setup_python_venv=setup_python_venv_steps,
        pip_config_index_url=(
            '' if with_gpu else 'RUN pip config set global.extra-index-url https://download.pytorch.org/whl/cpu'
        ),
        nvidia_env_vars=nvidia_env_vars,
        mlflow_version=MLFLOW_VERSION,
    )

    datamint_config = mlflow_model.flavors.get('datamint', {})
    if datamint_config:
        task_type = datamint_config.get('task_type', '')
        modes = datamint_config.get('supported_modes', [])
        modes_str = ",".join(modes) if isinstance(modes, list) else str(modes)
        settings = datamint_config.get('model_settings', {})
        need_gpu = str(settings.get('need_gpu', with_gpu)).lower()

        dockerfile += (
            f'\nLABEL datamint.task_type="{task_type}" \\\n'
            f'      datamint.supported_modes="{modes_str}" \\\n'
            f'      datamint.need_gpu="{need_gpu}"\n'
        )

    (Path(output_dir) / 'Dockerfile').write_text(dockerfile)
    logger.debug(f"Generated Dockerfile:\n{dockerfile}")

    return mlflow_model


def _podman_supports_platform_flag() -> bool:
    """Whether the local podman's API supports ``--platform`` (podman >= 4.0)."""
    from podman import PodmanClient

    try:
        client = PodmanClient()
        version_str = client.version().get("Version", "0.0.0")
    except Exception as e:
        raise RuntimeError(_PODMAN_NOT_RUNNING_MSG) from e
    return int(version_str.split(".")[0]) >= 4


def _run_podman_build(
    context_dir: str,
    image_name: str,
    log_callback: Callable[[str], None] | None = None,
) -> None:
    """Run ``podman build`` in context_dir, tagging the result as image_name.

    Build output streams line by line as it happens, and also
    passed to log_callback if one is given (so a caller can, say, log it to
    their own file instead of/as well as the console).

    Raises:
        RuntimeError: If podman isn't installed/running, or the build fails
            (with the last 50 output lines attached for debugging).
    """
    platform_option = ["--platform", "linux/amd64"] if _podman_supports_platform_flag() else []

    podman_runtime = os.environ.get("PODMAN_RUNTIME", "").strip().lower()
    runtime_option = [f"--runtime={podman_runtime}"] if podman_runtime in ("runc", "crun") else []

    command = ["podman", "build", "-t", image_name, "-f", "Dockerfile", *platform_option, *runtime_option, "."]
    logger.info(f"Running: {' '.join(command)} (in {context_dir})")

    try:
        proc = subprocess.Popen(
            command, cwd=context_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(_PODMAN_CLI_NOT_FOUND_MSG) from e

    tail_lines: list[str] = []
    for raw_line in proc.stdout:
        line = raw_line.rstrip()
        print(line)
        if log_callback is not None:
            log_callback(line)
        tail_lines.append(line)
        if len(tail_lines) > 50:
            tail_lines.pop(0)

    if proc.wait():
        raise RuntimeError("podman build failed.\n" + "\n".join(tail_lines))


_BYTES_PER_MB = 1024 * 1024


def _get_image_info(image_name: str, image_tag: str) -> dict:
    """Look up the built image's id and size via the podman API."""
    from podman import PodmanClient

    try:
        client = PodmanClient()
        image_ref = f"{image_name.lower()}:{image_tag}"
        image = client.images.get(image_ref)
        return {
            'image_id': image.short_id,
            'size_mb': round(image.attrs.get('Size', 0) / _BYTES_PER_MB, 2),
        }
    except Exception as e:
        logger.warning(f"Failed to get image info for {image_name}:{image_tag}: {e}")
        return {'image_id': None, 'size_mb': None}


@dataclass
class DockerRunResult:
    """A running local podman container, ready to be used with predict_local()."""
    container_id: str
    container_name: str
    host_port: int


def _wait_for_ready(container, host_port: int, timeout: float, interval: float = 2.0) -> None:
    """Poll the container's ``/ping`` endpoint until it responds with HTTP 200.

    Raises:
        RuntimeError: If the container exits before becoming ready, or the
            timeout is reached (with the last 50 log lines attached).
    """
    url = f"http://127.0.0.1:{host_port}/ping"
    start_time = time.monotonic()
    last_error: str | None = None

    while time.monotonic() - start_time < timeout:
        container.reload()
        if container.status == "exited":
            break
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                logger.info(f"Container {container.name} ready after {time.monotonic() - start_time:.1f}s")
                return
            last_error = f"HTTP {resp.status_code}"
        except requests.exceptions.RequestException as e:
            last_error = str(e)
        time.sleep(interval)

    logs = "\n".join(line.decode() if isinstance(line, bytes) else line
                      for line in container.logs(tail=50))
    if container.status == "exited":
        raise RuntimeError(f"Container {container.name} exited before becoming ready.\n{logs}")
    raise RuntimeError(
        f"Container {container.name} did not become ready within {timeout}s at {url}. "
        f"Last error: {last_error}\n{logs}"
    )


def run_docker_container(
    image_name: str,
    image_tag: str = DEFAULT_DEPLOY_MODEL_ALIAS,
    host_port: int = 8080,
    container_port: int = 8080,
    with_gpu: bool = False,
    container_name: str | None = None,
    wait_ready: bool = True,
    ready_timeout: float = 60.0,
) -> DockerRunResult:
    """Run a previously built local podman image so it can serve predictions.

    Args:
        image_name: Name of the image to run, as returned by
            :func:`build_docker_image` (``DockerBuildResult.image_name``).
        image_tag: Tag of the image to run.
        host_port: Host port to publish the container's serving port on.
            Pass this same value as ``container_port`` to
            :func:`datamint.mlflow.models.local_inference.predict_local`.
        container_port: Port the model server listens on inside the
            container (``8080``, the image's built-in default).
        with_gpu: Whether to grant the container GPU access via NVIDIA CDI
            (``nvidia.com/gpu=all``). Requires an image built with
            ``build_docker_image(..., with_gpu=True)`.
        container_name: Optional name for the container. Defaults to a
            podman-generated name.
        wait_ready: Poll the container's ``/ping`` endpoint until it
            responds, raising if it crashes or times out first.
        ready_timeout: Seconds to wait for readiness when ``wait_ready=True``.

    Returns:
        A :class:`DockerRunResult` with the running container's id/name and
        the host port to pass to ``predict_local()``.
    """
    from podman import PodmanClient
    from podman.errors import ImageNotFound

    full_image_name = f"{image_name.lower()}:{image_tag}"

    try:
        client = PodmanClient()
    except Exception as e:
        raise RuntimeError(_PODMAN_NOT_RUNNING_MSG) from e

    run_kwargs = dict(
        image=full_image_name,
        detach=True,
        ports={f"{container_port}/tcp": host_port},
    )
    if container_name is not None:
        run_kwargs["name"] = container_name
    if with_gpu:
        run_kwargs["devices"] = ["nvidia.com/gpu=all"]

    try:
        container = client.containers.run(**run_kwargs)
    except ImageNotFound as e:
        raise RuntimeError(
            f"Image '{full_image_name}' not found locally. Build it first with build_docker_image()."
        ) from e

    logger.info(f"Started container {container.name} ({container.id[:12]}) from {full_image_name}")

    if wait_ready:
        _wait_for_ready(container, host_port, timeout=ready_timeout)

    return DockerRunResult(container_id=container.id, container_name=container.name, host_port=host_port)


def stop_docker_container(
    container: DockerRunResult | str,
    timeout: int = 10,
    remove: bool = True,
) -> None:
    """Stop a container started with :func:`run_docker_container`.

    Args:
        container: The :class:`DockerRunResult` returned by
            ``run_docker_container()``, or a container id/name string.
        timeout: Seconds to wait for a graceful stop before killing it.
        remove: Whether to also remove the container after stopping it
            (``True`` by default, since these containers are meant to be
            disposable local dev/test instances).
    """
    from podman import PodmanClient
    from podman.errors import NotFound

    container_id = container.container_id if isinstance(container, DockerRunResult) else container

    try:
        client = PodmanClient()
    except Exception as e:
        raise RuntimeError(_PODMAN_NOT_RUNNING_MSG) from e

    try:
        handle = client.containers.get(container_id)
        handle.stop(timeout=timeout)
        logger.info(f"Stopped container {handle.name} ({handle.id[:12]})")
        if remove:
            handle.remove(force=True)
            logger.info(f"Removed container {handle.name} ({handle.id[:12]})")
    except NotFound:
        logger.warning(f"Container '{container_id}' not found, nothing to stop.")


def remove_docker_image(
    image_name: str,
    image_tag: str = DEFAULT_DEPLOY_MODEL_ALIAS,
    force: bool = False,
) -> None:
    """Remove a locally built podman image, as returned by :func:`build_docker_image`.

    Args:
        image_name: Name of the image to remove (``DockerBuildResult.image_name``).
        image_tag: Tag of the image to remove.
        force: Remove the image even if a (stopped) container still references it.
    """
    from podman import PodmanClient
    from podman.errors import ImageNotFound

    full_image_name = f"{image_name.lower()}:{image_tag}"

    try:
        client = PodmanClient()
    except Exception as e:
        raise RuntimeError(_PODMAN_NOT_RUNNING_MSG) from e

    try:
        client.images.remove(full_image_name, force=force)
        logger.info(f"Removed image {full_image_name}")
    except ImageNotFound:
        logger.warning(f"Image '{full_image_name}' not found, nothing to remove.")


def _sanitize_container_segment(segment: str, default_value: str) -> str:
    segment = segment.strip().lower()
    segment = re.sub(r"\s+", "_", segment)
    segment = re.sub(r"[^a-z0-9._-]", "-", segment)
    segment = segment.strip("._-")
    return segment or default_value


def sanitize_image_name(image_name: str) -> str:
    """Normalize a container image name to OCI-compatible format."""
    parts = [p for p in image_name.split("/") if p]
    if not parts:
        return "image"
    return "/".join(_sanitize_container_segment(part, "image") for part in parts)


def sanitize_image_tag(image_tag: str) -> str:
    """Normalize a container image tag to OCI-compatible format."""
    tag = image_tag.strip()
    tag = re.sub(r"\s+", "_", tag)
    tag = re.sub(r"[^a-zA-Z0-9._-]", "-", tag)
    tag = tag.strip("._-")
    return tag.lower() or "latest"


def _extract_image_name_from_model_uri(model_uri: str) -> str:
    """Derive an image name from a ``models:/<name>@<alias>`` URI, stripping the alias/version."""
    model_part = model_uri.split("/", 1)[1]  # "MyModel@champion"
    name_without_alias = model_part.split("@")[0]  # "MyModel"
    base_name = name_without_alias.split("/")[0]  # "MyModel"
    return sanitize_image_name(base_name)


@dataclass
class DockerBuildResult:
    """Result of a local podman image build. """
    image_name: str
    image_tag: str
    image_id: str | None
    size_mb: float | None


def build_docker_image(
    model_uri: str,
    image_name: str | None = None,
    image_tag: str = DEFAULT_DEPLOY_MODEL_ALIAS,
    with_gpu: bool = False,
    log_callback: Callable[[str], None] | None = None,
) -> DockerBuildResult:
    """Build a podman image for an MLflow model, entirely on the local machine.

    Downloads the model, generates a serving ``Dockerfile`` from it, and
    builds it with podman. Requires podman to be installed and running 
    locally (see :func:`_run_podman_build`).

    Args:
        model_uri: MLflow model URI, e.g. ``models:/MyModel@champion``.
        image_name: Name for the built image. Derived from model_uri if
            not given (only possible when *model_uri* starts with
            ``models:/``).
        image_tag: Tag for the built image, e.g. ``'v1'``, ``'champion'``.
        with_gpu: Whether to build a GPU (CUDA) capable image.
        log_callback: Optional callback invoked with each new build-log
            line as it streams in. Lines are always printed regardless.

    Returns:
        A :class:`DockerBuildResult` with the final image name/tag/id/size.

    Example:
        >>> result = build_docker_image("models:/MyModel@champion")
        >>> result.image_name, result.image_tag
        ('mymodel', 'champion')
    """
    if image_name is None:
        if not model_uri.startswith("models:/"):
            raise ValueError("image_name must be provided if model_uri does not start with 'models:/'.")
        image_name = _extract_image_name_from_model_uri(model_uri)
    else:
        image_name = sanitize_image_name(image_name)

    image_tag = sanitize_image_tag(image_tag)
    full_image_name = f"{image_name}:{image_tag}"
    logger.info(f"Building Docker image: {full_image_name}")

    with tempfile.TemporaryDirectory() as build_dir:
        _generate_dockerfile(model_uri, build_dir, with_gpu=with_gpu)
        _run_podman_build(build_dir, full_image_name, log_callback=log_callback)

    info = _get_image_info(image_name, image_tag)
    return DockerBuildResult(
        image_name=image_name,
        image_tag=image_tag,
        image_id=info['image_id'],
        size_mb=info['size_mb'],
    )
