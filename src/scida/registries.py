"""
This module contains registries for dataset and dataseries subclasses.
Subclasses are automatically registered through __init_subclass__
"""

from typing import Dict, Type

dataset_type_registry: Dict[str, Type] = {}
dataseries_type_registry: Dict[str, Type] = {}
