"""DTOs for point/box/text prompts sent to promptable segmentation models (e.g. SAM3)."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PointPrompt(BaseModel):
    """A single point prompt, in original image pixel space.

    Attributes:
        label: ``1`` for foreground, ``0`` for background.
        x: X coordinate.
        y: Y coordinate.
    """

    model_config = ConfigDict(extra='ignore')

    label: int = Field(..., description="1 for foreground, 0 for background")
    x: float = Field(..., description="X coordinate, in original image pixel space")
    y: float = Field(..., description="Y coordinate, in original image pixel space")

    @model_validator(mode='after')
    def _validate_label(self) -> 'PointPrompt':
        if self.label not in (0, 1):
            raise ValueError("label must be 0 (background) or 1 (foreground)")
        return self


class BoxPrompt(BaseModel):
    """A bounding box prompt, in original image pixel space."""

    model_config = ConfigDict(extra='ignore')

    x_min: float = Field(..., description="Minimum X coordinate")
    y_min: float = Field(..., description="Minimum Y coordinate")
    x_max: float = Field(..., description="Maximum X coordinate")
    y_max: float = Field(..., description="Maximum Y coordinate")

    @model_validator(mode='after')
    def _validate_box_coordinates(self) -> 'BoxPrompt':
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")
        if self.y_max <= self.y_min:
            raise ValueError("y_max must be greater than y_min")
        return self


class InferencePrompts(BaseModel):
    """Prompts for promptable segmentation models (e.g. SAM3-based).

    At least one of ``text``, ``points``, or ``boxes`` must be provided. Whether a
    given deployed model actually uses the prompts is up to that model; a model that
    doesn't support prompts simply ignores them.
    """

    model_config = ConfigDict(extra='ignore')

    text: str | None = Field(None, description="Concept phrase, e.g. 'object'")
    points: list[PointPrompt] | None = Field(None, description="List of point prompts")
    boxes: list[BoxPrompt] | None = Field(None, description="List of box prompts")

    @model_validator(mode='after')
    def _validate_non_empty(self) -> 'InferencePrompts':
        if not self.text and not self.points and not self.boxes:
            raise ValueError("At least one of text, points, or boxes must be provided")
        return self

    def to_dict(self) -> dict:
        """Return a JSON-compatible dict with only the set fields."""
        return self.model_dump(exclude_none=True)
