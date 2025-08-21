"""
Hardware-specific BSP configurations.

This module provides specialized BSP hardware configurations for different architectures.
"""

from .graphcore import GraphcoreIPUHardware, create_ipu_hardware

__all__ = ['GraphcoreIPUHardware', 'create_ipu_hardware']