'''建议引擎'''
from typing import List
from .base import BaseHarvester
from .term_alignment import TermAlignmentHarvester
from .weight_negotiation import WeightNegotiationHarvester
from ..report import HarmonyReport

class SuggestionEngine:
    def __init__(self, harvesters: List[BaseHarvester] = None):
        self.harvesters = harvesters or [
            TermAlignmentHarvester(),
            WeightNegotiationHarvester(),
        ]

    def register_harvester(self, harvester: BaseHarvester):
        self.harvesters.append(harvester)

    def generate_suggestions(self, report: HarmonyReport) -> List[str]:
        all_suggestions = []
        for harvester in self.harvesters:
            if harvester.analyze(report):
                all_suggestions.extend(harvester.generate_intervention(report))
        return all_suggestions
