"""DTO for save-results options when submitting inference jobs."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SaveResultsOptions(BaseModel):
    """Configuration for saving inference results as annotations on Datamint.

    These options are only applied when ``save_results=True`` on the prediction request.

    Attributes:
        worklist_id: Annotation worklist ID to associate saved annotations with.
        annotation_source: Source tag for saved annotations. Use ``'model_deploy'`` for
            deployed model inference or ``'model_pipeline'`` for pipeline-based runs.
        imported_from: Free-text describing the origin or context of this inference run.
        author_email: Email to attribute as the author of saved annotations. Defaults to
            the API key owner if not provided.
    """

    model_config = ConfigDict(extra='ignore')

    worklist_id: str | None = Field(
        default=None,
        description="Annotation worklist ID to associate saved annotations with.",
    )
    annotation_source: Literal['model_pipeline', 'model_deploy'] | None = Field(
        default='model_deploy',
        description=(
            "Source tag for saved annotations. Use 'model_deploy' for deployed model "
            "inference or 'model_pipeline' for pipeline-based runs."
        ),
    )
    imported_from: str | None = Field(
        default=None,
        description="Free-text describing the origin or context of this inference run.",
    )
    author_email: str | None = Field(
        default=None,
        pattern=r'^[\w.+\-]+@[\w\-]+\.[\w.\-]+$',
        description=(
            "Email to attribute as the author of saved annotations. "
            "Defaults to the API key owner if not provided."
        ),
    )

    def to_dict(self) -> dict[str, str]:
        """Return a dict with only the set (non-``None``) fields."""
        return {k: v for k, v in self.model_dump().items() if v is not None}
