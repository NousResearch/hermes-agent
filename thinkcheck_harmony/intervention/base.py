'''矛盾捕获器抽象基类（预留扩展钩子）'''
from abc import ABC, abstractmethod
from typing import List
from ..report import HarmonyReport

class BaseHarvester(ABC):
    @abstractmethod
    def analyze(self, report: HarmonyReport) -> bool:
        pass

    @abstractmethod
    def generate_intervention(self, report: HarmonyReport) -> List[str]:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    def probe_field(self, report: HarmonyReport) -> dict:
        return {}

    def get_pattern_library(self) -> list:
        return []
