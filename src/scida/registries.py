"""
This module contains registries for dataset and dataseries subclasses.
Subclasses are automatically registered through __init_subclass__
"""

from __future__ import annotations

dataset_type_registry: dict[str, type] = {}
dataseries_type_registry: dict[str, type] = {}
mixin_type_registry: dict[str, type] = {}
