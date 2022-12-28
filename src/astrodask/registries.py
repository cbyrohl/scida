# registries for different object types. mostly automatically registered through __init_subclass__
from typing import Dict, Type

dataset_type_registry: Dict[str, Type] = {}
