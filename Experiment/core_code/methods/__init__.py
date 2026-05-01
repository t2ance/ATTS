"""Method registry: schema layer (specs) + behavior layer (registry)."""
from methods.specs import MethodSpec
from methods.registry import MethodConfig, METHODS, get_method

__all__ = ["MethodSpec", "MethodConfig", "METHODS", "get_method"]
