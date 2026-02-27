import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import importlib.util

logger = logging.getLogger(__name__)

# Try to import qdrant_client, but make it optional
QDRANT_AVAILABLE = False
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    QDRANT_AVAILABLE = True
except ImportError:
    logger.warning("qdrant-client not installed. Vector search will be disabled.")

# Load Part model directly
_parts_models_path = Path(__file__).parent / "models.py"
spec = importlib.util.spec_from_file_location("parts_models", _parts_models_path)
_parts_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_parts_models)

Part = _parts_models.Part


class PartsVectorStore:
    """
    Vector store for semantic search over Dynamic Parts.
    Uses Qdrant for vector similarity search.
    """
    
    COLLECTION_NAME = "hermes_parts"
    
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        self.client = None
        self.embedding_model = None
        
        if not QDRANT_AVAILABLE:
            logger.warning("Qdrant not available. Vector search disabled.")
            return
        
        try:
            self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
            self._ensure_collection()
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant: {e}")
            self.client = None
    
    def _ensure_collection(self):
        if not self.client:
            return
        
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.COLLECTION_NAME not in collection_names:
                self.client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=384,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.COLLECTION_NAME}")
        except Exception as e:
            logger.warning(f"Could not ensure Qdrant collection: {e}")
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text using available embedding models."""
        if self.embedding_model:
            return self.embedding_model.encode(text).tolist()
        
        # Try to use sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            return self.embedding_model.encode(text).tolist()
        except ImportError:
            pass
        
        # Try OpenAI embeddings
        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception:
            pass
        
        return None
    
    def _get_text_for_embedding(self, part: Part) -> str:
        """Combine relevant part fields for embedding."""
        text_parts = [
            part.name,
            part.description,
            " ".join(part.triggers),
            " ".join(part.wants),
            " ".join(part.phrases),
            part.personality,
            part.emotion,
        ]
        return " | ".join([t for t in text_parts if t])
    
    def index_part(self, part: Part) -> bool:
        """Index a single part for vector search."""
        if not self.client:
            return False
        
        try:
            text = self._get_text_for_embedding(part)
            embedding = self._get_embedding(text)
            
            if not embedding:
                logger.warning(f"Could not get embedding for part: {part.name}")
                return False
            
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=part.id,
                        vector=embedding,
                        payload={
                            "part_id": part.id,
                            "name": part.name,
                            "description": part.description,
                            "triggers": part.triggers,
                            "wants": part.wants,
                            "archived": part.archived,
                        }
                    )
                ]
            )
            return True
        except Exception as e:
            logger.error(f"Error indexing part {part.name}: {e}")
            return False
    
    def index_parts(self, parts: List[Part]) -> int:
        """Index multiple parts. Returns count of successfully indexed."""
        if not self.client:
            return 0
        
        indexed = 0
        for part in parts:
            if not part.archived:
                if self.index_part(part):
                    indexed += 1
        return indexed
    
    def search(self, query: str, limit: int = 5, include_archived: bool = False) -> List[Dict[str, Any]]:
        """Search parts by semantic similarity."""
        if not self.client:
            return []
        
        try:
            embedding = self._get_embedding(query)
            if not embedding:
                return []
            
            search_filters = None
            if not include_archived:
                search_filters = Filter(
                    must_not=[
                        FieldCondition(key="archived", match=MatchValue(value=True))
                    ]
                )
            
            results = self.client.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=embedding,
                limit=limit,
                query_filter=search_filters,
                with_payload=True
            )
            
            return [
                {
                    "part_id": r.payload["part_id"],
                    "name": r.payload["name"],
                    "description": r.payload.get("description", ""),
                    "score": r.score,
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Error searching parts: {e}")
            return []
    
    def delete_part(self, part_id: str) -> bool:
        """Remove a part from the vector index."""
        if not self.client:
            return False
        
        try:
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=[part_id]
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting part from index: {e}")
            return False
    
    def rebuild_index(self, parts: List[Part]) -> int:
        """Clear and rebuild the entire index."""
        if not self.client:
            return 0
        
        try:
            # Delete collection
            try:
                self.client.delete_collection(collection_name=self.COLLECTION_NAME)
            except:
                pass
            
            # Recreate
            self._ensure_collection()
            
            # Re-index all parts
            return self.index_parts(parts)
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            return 0


def get_vector_store() -> PartsVectorStore:
    """Get or create the vector store instance."""
    return PartsVectorStore()
