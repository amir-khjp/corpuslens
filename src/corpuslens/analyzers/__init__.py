from corpuslens.analyzers.base import Analyzer, AnalyzerResult, Chart
from corpuslens.analyzers.basic_stats import BasicStatsAnalyzer
from corpuslens.analyzers.clustering import ClusteringAnalyzer
from corpuslens.analyzers.duplicates import DuplicateAnalyzer
from corpuslens.analyzers.keywords import KeywordAnalyzer
from corpuslens.analyzers.labels import LabelBalanceAnalyzer
from corpuslens.analyzers.pii import PiiAnalyzer

DEFAULT_ANALYZERS = [
    BasicStatsAnalyzer(),
    DuplicateAnalyzer(),
    PiiAnalyzer(),
    KeywordAnalyzer(),
    ClusteringAnalyzer(),
    LabelBalanceAnalyzer(),
]

__all__ = [
    "DEFAULT_ANALYZERS",
    "Analyzer",
    "AnalyzerResult",
    "BasicStatsAnalyzer",
    "Chart",
    "ClusteringAnalyzer",
    "DuplicateAnalyzer",
    "KeywordAnalyzer",
    "LabelBalanceAnalyzer",
    "PiiAnalyzer",
]
