"""Locomotion environments for legged robots."""

# 显式导入 robots 模块以确保环境注册 / Explicitly import robots module to ensure environments are registered
from . import robots

# 确保所有模块在导入时都被加载 / Ensure all modules are loaded on import
__all__ = ["robots"]

