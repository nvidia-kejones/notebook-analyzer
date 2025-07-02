"""
Notebook Analyzer Package

Enhanced static imports for the notebook analyzer functionality with
comprehensive NVIDIA Best Practices integration.
"""

from .core import GPUAnalyzer, GPURequirement, LLMAnalyzer, NVIDIABestPracticesLoader

__all__ = ['GPUAnalyzer', 'GPURequirement', 'LLMAnalyzer', 'NVIDIABestPracticesLoader']
__version__ = '3.1.0' 