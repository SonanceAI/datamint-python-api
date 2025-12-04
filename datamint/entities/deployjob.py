from datamint.entities.base_entity import BaseEntity


class DeployJob(BaseEntity):
    id: str
    status: str
    model_name: str
    model_version: int | None = None
    model_alias: str | None = None
    image_name: str | None = None
    image_tag: str | None = None
    error_message: str | None = None
    progress_percentage: int = 0
    current_step: str | None = None
    with_gpu: bool = False
    recent_logs: list[str] | None = None
    started_at: str | None = None
    completed_at: str | None = None