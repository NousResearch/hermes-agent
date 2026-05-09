'''概念关系图：关系本体论的工程实现'''
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .utils.text_processing import split_sentences
from .utils.embedding import get_embeddings
from .config import DRIFT_THRESHOLD

class ConceptGraph:
    def __init__(self, text: str, key_terms: list):
        self.text = text
        self.key_terms = key_terms
        self.sentences = split_sentences(text)
        self.term_occurrences = self._locate_terms()
        self.consistency_scores = self._compute_consistencies()

    def _locate_terms(self) -> dict:
        locations = {}
        for term in self.key_terms:
            positions = [i for i, s in enumerate(self.sentences) if term in s]
            if positions:
                locations[term] = positions
        return locations

    def _compute_consistencies(self) -> dict:
        scores = {}
        for term, positions in self.term_occurrences.items():
            if len(positions) < 2:
                scores[term] = 1.0
                continue
            term_sentences = [self.sentences[i] for i in positions]
            embeddings = get_embeddings(term_sentences)
            sim_matrix = cosine_similarity(embeddings)
            triu_indices = np.triu_indices_from(sim_matrix, k=1)
            if len(triu_indices[0]) == 0:
                scores[term] = 1.0
            else:
                scores[term] = float(np.mean(sim_matrix[triu_indices]))
        return scores

    def get_avg_consistency(self) -> float:
        if not self.consistency_scores:
            return 1.0
        return float(np.mean(list(self.consistency_scores.values())))

    def get_first_occurrence_positions(self) -> list:
        first_positions = []
        for positions in self.term_occurrences.values():
            if positions:
                first_positions.append(positions[0])
        return sorted(first_positions)

    def get_drift_warnings(self) -> list:
        warnings = []
        for term, score in self.consistency_scores.items():
            if score < DRIFT_THRESHOLD:
                warnings.append({
                    "term": term,
                    "consistency": round(score, 3),
                    "threshold": DRIFT_THRESHOLD,
                    "occurrences": len(self.term_occurrences[term]),
                    "sentences": [self.sentences[i] for i in self.term_occurrences[term]]
                })
        return warnings
