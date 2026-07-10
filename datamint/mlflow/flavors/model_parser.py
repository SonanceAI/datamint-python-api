"""Utility functions for parsing model references.

This module provides helpers for working with the Datamint model reference
format that uses two separators:

* ``/`` (slash) — separates **customer** prefix, e.g. ``customer_name/model_ref``
* ``:`` (colon) — separates **project** prefix from model name within same-customer scope, e.g. ``project:model``

Supported formats (from most specific to least):

.. code-block::

    customer/model              →  cross-customer reference (model is unique per customer)
    project:model               →  same-customer with project prefix
    model                       →  bare model name (same-customer, any project)
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, eq=True)
class ModelReference:
    """Parsed model reference with named attributes.

    Attributes:
        customer_name: Customer identifier when slash was used (cross-customer).
            ``None`` for same-customer references.
        project_name: Project identifier when colon was used (same-customer prefix).
            ``None`` for bare names and cross-customer references.
        model_name: The bare model name, extracted from any format.
    """

    customer_name: str | None
    project_name: str | None
    model_name: str


def parse_model_reference(reference: str) -> ModelReference:
    """Parse a human-readable model reference into a :class:`ModelReference` dataclass.

    Supports three slash/colon-delimited formats:

    * ``model`` — bare model name → ``ModelReference(None, None, 'model')``
    * ``project:model`` — same-customer with project prefix →
      ``ModelReference(None, 'project', 'model')``
    * ``customer/model`` — cross-customer reference →
      ``ModelReference('customer', None, 'model')``

    **Separator semantics:**

    * ``/`` (slash) indicates a **cross-customer** reference. The left part is the
      customer name; since model names are unique within a customer, no project
      specification is needed.
    * ``:`` (colon) indicates a **same-customer project prefix**. The left part is
      the project name within the current customer's scope.

    **Parsing order:** The function checks for ``/`` first to determine if this
    is a cross-customer reference. If no slash is found, it checks for ``:`` to
    determine if there's a same-customer project prefix.

    Args:
        reference: A string in one of the supported formats described above.

    Returns:
        A :class:`ModelReference` dataclass with named attributes
        ``customer_name``, ``project_name``, and ``model_name``.

    Raises:
        ValueError: If the reference is empty or malformed.
    """
    if not reference or not reference.strip():
        raise ValueError("Model reference cannot be empty")

    reference = reference.strip()

    customer_name: str | None = None
    project_name: str | None = None
    model_name: str

    # Step 1: Check for slash — cross-customer reference (customer/model)
    if "/" in reference:
        slash_parts = reference.split("/", 1)
        customer_name = slash_parts[0].strip()
        model_name = slash_parts[1].strip() if len(slash_parts) > 1 else ""

        if not model_name:
            raise ValueError(
                f"Invalid model reference '{reference}'. Expected format: 'customer/model' "
                "or just 'model_name'"
            )

    # Step 2: Check for colon — same-customer project prefix (project:model)
    elif ":" in reference:
        colon_parts = reference.split(":", 1)
        project_name = colon_parts[0].strip()
        model_name = colon_parts[1].strip() if len(colon_parts) > 1 else ""

        if not model_name:
            raise ValueError(
                f"Invalid model reference '{reference}'. Expected format: 'project:model' "
                "or just 'model'"
            )

    else:
        # Bare model name
        model_name = reference

    if not model_name:
        raise ValueError(
            f"Invalid model reference '{reference}'. Model name cannot be empty."
        )

    return ModelReference(
        customer_name=customer_name,
        project_name=project_name,
        model_name=model_name,
    )


def extract_model_name(reference: str) -> str:
    """Extract the bare model name from a slash/colon-delimited reference.

    Args:
        reference: A string in one of the supported formats:

            * ``my-model`` → ``'my-model'``
            * ``project:model`` → ``'model'`` (same-customer project prefix)
            * ``customer/model`` → ``'model'`` (cross-customer reference)

    Returns:
        The bare model name with no prefix.

    Example:
        >>> extract_model_name("hospital-a/chest-xray")
        'chest-xray'
        >>> extract_model_name("vision-project:resnet50")
        'resnet50'
        >>> extract_model_name("my-model")
        'my-model'
    """
    return parse_model_reference(reference).model_name
