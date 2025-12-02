"""
GO Extractor package initialization.
"""

from .main import SimpleGOExtractor, extract_go_embeddings, analyze_embeddings, extract_simple_comparison

__all__ = ['SimpleGOExtractor', 'extract_go_embeddings', 'analyze_embeddings', 'extract_simple_comparison']