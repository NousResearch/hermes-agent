'''嵌入模型封装（带缓存）'''
from sentence_transformers import SentenceTransformer
from ..config import EMBEDDING_MODEL

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def get_embeddings(sentences):
    model = get_model()
    return model.encode(sentences)
