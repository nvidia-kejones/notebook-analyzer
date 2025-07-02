"""
Notebook Analyzer Package

Safe static imports for the notebook analyzer functionality.
"""

from .core import GPUAnalyzer, GPURequirement, LLMAnalyzer

__all__ = ['GPUAnalyzer', 'GPURequirement', 'LLMAnalyzer']
__version__ = '3.0.0' 