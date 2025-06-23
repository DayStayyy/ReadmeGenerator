#!/usr/bin/env python3
"""
Système d'Embedding et Base Vectorielle pour RAG
Transforme les chunks en vecteurs et permet la recherche sémantique
"""

import numpy as np
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import pickle
import os
from pathlib import Path

# Pour l'embedding local gratuit
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers non installé. Utilise: pip install sentence-transformers")

# Pour la base vectorielle simple
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  faiss non installé. Utilise: pip install faiss-cpu")


@dataclass
class EmbeddedChunk:
    """Chunk avec son embedding"""
    chunk_id: str
    content: str
    chunk_type: str
    file_path: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None
    start_line: int = 0
    end_line: int = 0


class CodeEmbedder:
    """Gestionnaire d'embeddings spécialisé pour le code"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Modèles recommandés gratuits :
        - all-MiniLM-L6-v2 : Rapide, bon généraliste (384 dim)
        - all-mpnet-base-v2 : Meilleur qualité (768 dim)
        - microsoft/codebert-base : Spécialisé code (768 dim)
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print(f"🔄 Chargement du modèle d'embedding: {model_name}")
                self.model = SentenceTransformer(model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                print(f"✅ Modèle chargé! Dimension: {self.embedding_dim}")
            except Exception as e:
                print(f"❌ Erreur chargement modèle: {e}")
                self.model = None
        else:
            print("❌ sentence-transformers requis pour l'embedding local")

    def embed_chunks(self, chunks: List) -> List[EmbeddedChunk]:
        """Transforme une liste de chunks en chunks avec embeddings"""
        if not self.model:
            print("❌ Pas de modèle d'embedding disponible")
            return []

        embedded_chunks = []

        print(f"🔄 Création des embeddings pour {len(chunks)} chunks...")

        # Préparer les textes pour l'embedding
        texts_to_embed = []
        for chunk in chunks:
            # Créer un texte enrichi pour l'embedding
            enhanced_text = self._create_enhanced_text(chunk)
            texts_to_embed.append(enhanced_text)

        try:
            # Créer tous les embeddings en une fois (plus efficace)
            embeddings = self.model.encode(texts_to_embed, show_progress_bar=True)

            # Associer chaque embedding à son chunk
            for i, chunk in enumerate(chunks):
                embedded_chunk = EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    chunk_type=chunk.chunk_type,
                    file_path=chunk.file_path,
                    metadata=chunk.metadata,
                    embedding=embeddings[i],
                    start_line=chunk.start_line,
                    end_line=chunk.end_line
                )
                embedded_chunks.append(embedded_chunk)

            print(f"✅ {len(embedded_chunks)} embeddings créés!")

        except Exception as e:
            print(f"❌ Erreur lors de l'embedding: {e}")
            return []

        return embedded_chunks

    def _create_enhanced_text(self, chunk) -> str:
        """Crée un texte enrichi pour un meilleur embedding"""

        # Base: le contenu du chunk
        enhanced_parts = [chunk.content]

        # Ajouter le contexte du type de chunk
        type_context = {
            'class_definition': 'This is a Python class definition:',
            'module_header': 'This is a Python module header with imports:',
            'main_execution': 'This is the main execution block of a Python script:',
            'notebook_code_cell': 'This is a Jupyter notebook code cell:',
            'notebook_documentation': 'This is Jupyter notebook documentation:',
            'markdown_section': 'This is a markdown documentation section:',
            'configuration': 'This is a configuration file:',
            'project_dependencies': 'This is a project dependencies file:',
            'readme_documentation': 'This is README documentation:'
        }

        if chunk.chunk_type in type_context:
            enhanced_parts.insert(0, type_context[chunk.chunk_type])

        # Ajouter des métadonnées utiles
        metadata_parts = []

        if 'class_name' in chunk.metadata:
            metadata_parts.append(f"Class name: {chunk.metadata['class_name']}")

        if 'methods' in chunk.metadata:
            methods = chunk.metadata['methods']
            if methods:
                metadata_parts.append(f"Methods: {', '.join(methods)}")

        if 'header_title' in chunk.metadata:
            metadata_parts.append(f"Section: {chunk.metadata['header_title']}")

        if 'dependencies' in chunk.metadata:
            deps = chunk.metadata['dependencies']
            if deps:
                metadata_parts.append(f"Dependencies: {', '.join(deps[:5])}")  # Top 5

        # Ajouter le nom du fichier pour le contexte
        file_context = f"File: {chunk.file_path}"
        metadata_parts.append(file_context)

        if metadata_parts:
            enhanced_parts.insert(-1, " | ".join(metadata_parts))

        return "\n".join(enhanced_parts)

    def embed_query(self, query: str) -> Optional[np.ndarray]:
        """Crée l'embedding d'une query de recherche"""
        if not self.model:
            return None

        try:
            return self.model.encode([query])[0]
        except Exception as e:
            print(f"❌ Erreur embedding query: {e}")
            return None


class VectorDatabase:
    """Base de données vectorielle simple avec FAISS"""

    def __init__(self):
        self.index = None
        self.chunks = []
        self.embedding_dim = None

    def build_index(self, embedded_chunks: List[EmbeddedChunk]):
        """Construit l'index FAISS à partir des chunks embeddings"""

        if not embedded_chunks:
            print("❌ Aucun chunk avec embedding fourni")
            return

        if not FAISS_AVAILABLE:
            print("❌ FAISS requis pour la base vectorielle")
            return

        # Extraire les embeddings
        embeddings = np.array([chunk.embedding for chunk in embedded_chunks])
        self.embedding_dim = embeddings.shape[1]
        self.chunks = embedded_chunks

        print(f"🔄 Construction de l'index FAISS...")
        print(f"   - {len(embedded_chunks)} chunks")
        print(f"   - Dimension: {self.embedding_dim}")

        try:
            # Créer l'index FAISS (IndexFlatIP pour cosine similarity)
            self.index = faiss.IndexFlatIP(self.embedding_dim)

            # Normaliser les embeddings pour cosine similarity
            faiss.normalize_L2(embeddings)

            # Ajouter à l'index
            self.index.add(embeddings.astype('float32'))

            print(f"✅ Index construit! {self.index.ntotal} vecteurs indexés")

        except Exception as e:
            print(f"❌ Erreur construction index: {e}")
            self.index = None

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[EmbeddedChunk, float]]:
        """Recherche les k chunks les plus similaires"""

        if not self.index or query_embedding is None:
            return []

        try:
            # Normaliser la query
            query_normalized = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_normalized)

            # Rechercher
            scores, indices = self.index.search(query_normalized, k)

            # Retourner les résultats
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.chunks):  # Vérification sécurité
                    results.append((self.chunks[idx], float(score)))

            return results

        except Exception as e:
            print(f"❌ Erreur recherche: {e}")
            return []

    def search_by_query(self, query: str, embedder: CodeEmbedder, k: int = 5) -> List[Tuple[EmbeddedChunk, float]]:
        """Recherche par query texte"""
        query_embedding = embedder.embed_query(query)
        return self.search(query_embedding, k)

    def save(self, filepath: str):
        """Sauvegarde la base vectorielle"""
        if not self.index:
            print("❌ Pas d'index à sauvegarder")
            return

        try:
            # Sauvegarder l'index FAISS
            faiss.write_index(self.index, f"{filepath}.faiss")

            # Sauvegarder les chunks (sans embeddings pour économiser l'espace)
            chunks_data = []
            for chunk in self.chunks:
                chunk_dict = asdict(chunk)
                chunk_dict['embedding'] = None  # Pas besoin de resauvegarder
                chunks_data.append(chunk_dict)

            with open(f"{filepath}.json", 'w', encoding='utf-8') as f:
                json.dump({
                    'chunks': chunks_data,
                    'embedding_dim': self.embedding_dim,
                    'total_chunks': len(self.chunks)
                }, f, indent=2, ensure_ascii=False)

            print(f"✅ Base vectorielle sauvegardée: {filepath}.faiss + {filepath}.json")

        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")

    def load(self, filepath: str, embedder: CodeEmbedder):
        """Charge une base vectorielle sauvegardée"""
        try:
            if not FAISS_AVAILABLE:
                print("❌ FAISS requis pour charger")
                return False

            # Charger l'index FAISS
            self.index = faiss.read_index(f"{filepath}.faiss")

            # Charger les métadonnées
            with open(f"{filepath}.json", 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Reconstruire les chunks (sans embeddings)
            self.chunks = []
            for chunk_data in data['chunks']:
                chunk = EmbeddedChunk(**chunk_data)
                self.chunks.append(chunk)

            self.embedding_dim = data['embedding_dim']

            print(f"✅ Base vectorielle chargée: {len(self.chunks)} chunks")
            return True

        except Exception as e:
            print(f"❌ Erreur chargement: {e}")
            return False


class RAGRetriever:
    """Système de récupération pour RAG"""

    def __init__(self, embedder: CodeEmbedder, vector_db: VectorDatabase):
        self.embedder = embedder
        self.vector_db = vector_db

    def retrieve_for_query(self, query: str, k: int = 5, filter_types: List[str] = None) -> List[Dict]:
        """Récupère les chunks pertinents pour une query"""

        # Recherche vectorielle
        results = self.vector_db.search_by_query(query, self.embedder, k=k * 2)  # Plus large pour filtrer

        # Filtrer par type si demandé
        if filter_types:
            results = [(chunk, score) for chunk, score in results
                       if chunk.chunk_type in filter_types]

        # Limiter au k demandé
        results = results[:k]

        # Formater pour utilisation
        formatted_results = []
        for chunk, score in results:
            formatted_results.append({
                'content': chunk.content,
                'file_path': chunk.file_path,
                'chunk_type': chunk.chunk_type,
                'metadata': chunk.metadata,
                'similarity_score': score,
                'chunk_id': chunk.chunk_id
            })

        return formatted_results

    def retrieve_context_for_readme(self, question_type: str) -> List[Dict]:
        """Récupère le contexte spécialisé pour différents types de questions README"""

        queries_and_filters = {
            'project_description': {
                'query': 'main purpose goal description what does this project do',
                'filters': ['module_header', 'readme_documentation', 'class_definition'],
                'k': 3
            },
            'installation': {
                'query': 'install setup requirements dependencies how to install',
                'filters': ['project_dependencies', 'setup_script', 'configuration'],
                'k': 5
            },
            'usage_examples': {
                'query': 'example usage how to use main function entry point',
                'filters': ['main_execution', 'notebook_code_cell', 'standalone_functions'],
                'k': 4
            },
            'project_structure': {
                'query': 'structure architecture classes modules components',
                'filters': ['class_definition', 'module_header'],
                'k': 6
            }
        }

        if question_type not in queries_and_filters:
            return []

        config = queries_and_filters[question_type]
        return self.retrieve_for_query(
            query=config['query'],
            k=config['k'],
            filter_types=config['filters']
        )


# Test du système complet
if __name__ == "__main__":
    print("🧪 Test du système d'embedding et base vectorielle")

    # Simuler des chunks pour test
    from IntelligentChunker import Chunk

    test_chunks = [
        Chunk(
            content="import pandas as pd\nimport numpy as np",
            chunk_type="module_header",
            file_path="data_processor.py",
            metadata={"imports": ["pandas", "numpy"]},
            chunk_id="test_001"
        ),
        Chunk(
            content="class DataProcessor:\n    def __init__(self):\n        self.data = None",
            chunk_type="class_definition",
            file_path="data_processor.py",
            metadata={"class_name": "DataProcessor"},
            chunk_id="test_002"
        )
    ]

    if SENTENCE_TRANSFORMERS_AVAILABLE and FAISS_AVAILABLE:
        # Test complet
        embedder = CodeEmbedder("all-MiniLM-L6-v2")

        if embedder.model:
            embedded_chunks = embedder.embed_chunks(test_chunks)

            vector_db = VectorDatabase()
            vector_db.build_index(embedded_chunks)

            retriever = RAGRetriever(embedder, vector_db)

            # Test de recherche
            results = retriever.retrieve_for_query("Python data processing class", k=2)

            print(f"\n🔍 Résultats de recherche:")
            for result in results:
                print(f"  - {result['chunk_type']} (score: {result['similarity_score']:.3f})")
                print(f"    {result['content'][:100]}...")

            print(f"\n✅ Test réussi!")
    else:
        print("❌ Dépendances manquantes pour le test complet")
        print("   pip install sentence-transformers faiss-cpu")