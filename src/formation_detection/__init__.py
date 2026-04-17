"""
formation_detection — Classify team formations from player positions.



Modules:
    goalkeeper_filter.py     — Identify and exclude goalkeepers
    line_clustering.py       — Group outfield players into defensive lines
    template_matching.py     — Match detected lines to known formations
    temporal_smoother.py     — Average positions over time window
    formation_classifier.py  — Main formation detection logic
"""

from .goalkeeper_filter import GoalkeeperFilter
from .line_clustering import LineClustering
from .template_matching import TemplateMatcher
from .temporal_smoother import TemporalSmoother
from .formation_classifier import FormationClassifier