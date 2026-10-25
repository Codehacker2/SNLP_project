"""Optional document analysis; the original Digitizer and schema are unchanged."""

from .analyzer import DocumentAnalyzer
from .figures import FigureDetector
from .ner import NERUnavailableError, SpacyNER

__all__ = ["DocumentAnalyzer", "FigureDetector", "SpacyNER", "NERUnavailableError"]
