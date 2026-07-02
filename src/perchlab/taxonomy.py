"""Class-label helpers.

Perch V2's class names follow the iNaturalist taxonomy and are effectively
scientific names (e.g. ``Ardea cinerea``). The existing Raven tables the team
produces use the scientific name for both *Common Name* and *Species Code*, so
by default PerchLab does the same. This module is the seam where a richer
scientific -> common-name mapping can be plugged in later.
"""

from __future__ import annotations

from .logging import get_logger

_log = get_logger("taxonomy")


class TaxonomyMap:
    """Maps model class names to display/common names and species codes."""

    def __init__(self, class_names: list[str]) -> None:
        """Initialise from the model's ordered class names.

        Args:
            class_names: Class names aligned with the model's logits axis.
        """
        self.class_names = class_names

    def label(self, class_index: int) -> str:
        """Return the model label (scientific name) for a class index."""
        return self.class_names[class_index]

    def common_name(self, class_index: int) -> str:
        """Return a common name for a class index.

        Currently returns the scientific name (identity mapping); this is the
        documented v1 behaviour and matches the team's existing tables.
        """
        return self.class_names[class_index]

    def species_code(self, class_index: int) -> str:
        """Return the species code for a class index (scientific name in v1)."""
        return self.class_names[class_index]
