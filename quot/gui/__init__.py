"""
__init__.py

"""
import PySide6
import pyqtgraph

# Suppress annoying pointer dispatch warning
import os
os.environ['QT_LOGGING_RULES'] = 'qt.pointer.dispatch=false'

from .masker import reconstruct_mask
