#!/usr/bin/env python3
"""
Script d'intégration complet Scanner + Chunker + Embedding + RAG
Usage: python integration.py
"""

import sys
from pathlib import Path
import json

# Import des modules locaux
try:
    from ProjectScanner import ProjectScanner
    from IntelligentChunker import IntelligentChunker, Chunk
    from CodeEmbedder import CodeEmbedder, VectorDatabase, RAGRetriever
except ImportError as e:
    print(f"Erreur d'import: {e}")
    print("Assure-toi que tous les fichiers sont présents:")
    print("- ProjectScanner.py")
    print("- IntelligentChunker.py")
    print("- embedding_vectordb.py")
    sys.exit(1)


def main():
    """Fonction principale - Pipeline complet"""

    # 🎯 CONFIGURATION - Change le path ici !
    PROJECT_PATH = r"../VoitureGoBrrrr"  # 👈 CHANGE MOI !
    RAG_DB_NAME = "project_rag_db"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # ou "all-mpnet-base-v2" pour plus de qualité

    print("🚀 Pipeline complet Scanner → Chunker → Embedding → RAG")
    print("=" * 60)

    # 📁 PHASE 1: SCANNING
    print(f"\n📁 PHASE 1: Scanning du projet")
    print(f"Path: {PROJECT_PATH}")

    scanner = ProjectScanner()

    try:
        scan_result = scanner.scan_project(PROJECT_PATH)
        print(f"✅ Scan terminé!")
        print(f"   - Type de projet: {scan_result['project_type']}")
        print(f"   - Fichiers trouvés: {len(scan_result['all_files'])}")

        # Afficher la répartition par type
        handlers_count = {}
        for file_data in scan_result['all_files']:
            handler = file_data['handler']
            handlers_count[handler] = handlers_count.get(handler, 0) + 1

        print(f"   - Répartition par type:")
        for handler, count in sorted(handlers_count.items()):
            print(f"     * {handler}: {count}")

    except Exception as e:
        print(f"❌ Erreur lors du scan: {e}")
        return None

    # 🧩 PHASE 2: CHUNKING
    print(f"\n🧩 PHASE 2: Chunking intelligent")

    chunker = IntelligentChunker(max_chunk_size=1000)

    try:
        all_chunks = chunker.chunk_all_files(scan_result['all_files'])
        print(f"✅ Chunking terminé!")
        print(f"   - Total chunks: {len(all_chunks)}")

        # Afficher la répartition par type de chunk
        chunk_types_count = {}
        for chunk in all_chunks:
            chunk_type = chunk.chunk_type
            chunk_types_count[chunk_type] = chunk_types_count.get(chunk_type, 0) + 1

        print(f"   - Répartition par type de chunk:")
        for chunk_type, count in sorted(chunk_types_count.items()):
            print(f"     * {chunk_type}: {count}")

    except Exception as e:
        print(f"❌ Erreur lors du chunking: {e}")
        return None

    # 🧠 PHASE 3: EMBEDDING
    print(f"\n🧠 PHASE 3: Création des embeddings")
    print(f"Modèle: {EMBEDDING_MODEL}")

    try:
        embedder = CodeEmbedder(EMBEDDING_MODEL)

        if not embedder.model:
            print("❌ Impossible de charger le modèle d'embedding")
            print("   Installe: pip install sentence-transformers")
            return None

        embedded_chunks = embedder.embed_chunks(all_chunks)

        if not embedded_chunks:
            print("❌ Échec création des embeddings")
            return None

        print(f"✅ Embeddings créés!")
        print(f"   - Dimension: {embedder.embedding_dim}")
        print(f"   - Chunks embeddings: {len(embedded_chunks)}")

    except Exception as e:
        print(f"❌ Erreur lors de l'embedding: {e}")
        return None

    # 🗄️ PHASE 4: BASE VECTORIELLE
    print(f"\n🗄️ PHASE 4: Construction de la base vectorielle")

    try:
        vector_db = VectorDatabase()
        vector_db.build_index(embedded_chunks)

        if not vector_db.index:
            print("❌ Échec construction de l'index")
            print("   Installe: pip install faiss-cpu")
            return None

        print(f"✅ Base vectorielle construite!")
        print(f"   - Index FAISS: {vector_db.index.ntotal} vecteurs")

        # Sauvegarder la base
        vector_db.save(RAG_DB_NAME)

    except Exception as e:
        print(f"❌ Erreur construction base vectorielle: {e}")
        return None

    # 🔍 PHASE 5: TEST RAG
    print(f"\n🔍 PHASE 5: Test du système RAG")

    try:
        retriever = RAGRetriever(embedder, vector_db)

        # Tests de recherche
        test_queries = [
            "main entry point of the application",
            "class definitions and methods",
            "project dependencies and requirements",
            "configuration files and setup"
        ]

        print(f"Tests de recherche:")
        for query in test_queries:
            results = retriever.retrieve_for_query(query, k=2)
            print(f"\n   Query: '{query}'")
            print(f"   Résultats: {len(results)}")

            for i, result in enumerate(results[:1]):  # Premier résultat seulement
                print(f"     {i + 1}. {result['chunk_type']} dans {result['file_path']}")
                print(f"        Score: {result['similarity_score']:.3f}")
                print(f"        Preview: {result['content'][:80]}...")

        print(f"\n✅ RAG fonctionnel!")

    except Exception as e:
        print(f"❌ Erreur test RAG: {e}")
        return None

    # 📊 RÉSUMÉ FINAL
    print(f"\n" + "=" * 60)
    print(f"📊 RÉSUMÉ FINAL - SYSTÈME PRÊT!")
    print(f"   - Projet: {scan_result['project_info']['name']}")
    print(f"   - Type: {scan_result['project_type']}")
    print(f"   - Fichiers analysés: {len(scan_result['all_files'])}")
    print(f"   - Chunks générés: {len(all_chunks)}")
    print(f"   - Embeddings créés: {len(embedded_chunks)}")
    print(f"   - Base RAG: {RAG_DB_NAME}.faiss")
    print(f"   - Dépendances: {len(scan_result['dependencies']['requirements'])}")

    # Sauvegarder les résultats complets
    save_complete_results(scan_result, all_chunks, embedded_chunks)

    # Analyse pour README
    readme_analysis = analyze_project_for_readme(scan_result, all_chunks, retriever)
    print_readme_analysis(readme_analysis)

    return {
        'scan_result': scan_result,
        'chunks': all_chunks,
        'embedded_chunks': embedded_chunks,
        'retriever': retriever,
        'readme_analysis': readme_analysis
    }


def save_complete_results(scan_result, chunks, embedded_chunks):
    """Sauvegarde complète des résultats"""

    # Résumé du scan
    scan_summary = {
        'project_info': scan_result['project_info'],
        'project_type': scan_result['project_type'],
        'dependencies': scan_result['dependencies'],
        'files_count': len(scan_result['all_files']),
        'files_by_type': {}
    }

    # Compter par handler
    for file_data in scan_result['all_files']:
        handler = file_data['handler']
        scan_summary['files_by_type'][handler] = scan_summary['files_by_type'].get(handler, 0) + 1

    with open('scan_summary.json', 'w', encoding='utf-8') as f:
        json.dump(scan_summary, f, indent=2, ensure_ascii=False)

    # Résumé des chunks
    chunks_summary = {
        'total_chunks': len(chunks),
        'chunks_by_type': {},
        'chunks_details': []
    }

    for chunk in chunks:
        chunk_type = chunk.chunk_type
        chunks_summary['chunks_by_type'][chunk_type] = chunks_summary['chunks_by_type'].get(chunk_type, 0) + 1

        chunks_summary['chunks_details'].append({
            'chunk_id': chunk.chunk_id,
            'chunk_type': chunk.chunk_type,
            'file_path': chunk.file_path,
            'content_length': len(chunk.content),
            'metadata_keys': list(chunk.metadata.keys()),
            'content_preview': chunk.content[:150]
        })

    with open('chunks_summary.json', 'w', encoding='utf-8') as f:
        json.dump(chunks_summary, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Résultats sauvegardés:")
    print(f"   - scan_summary.json")
    print(f"   - chunks_summary.json")
    print(f"   - project_rag_db.faiss + project_rag_db.json")


def analyze_project_for_readme(scan_result, chunks, retriever):
    """Analyse spécialisée pour génération de README avec RAG"""

    analysis = {
        'project_overview': {},
        'main_components': {},
        'installation_info': {},
        'usage_examples': {},
        'project_structure': {}
    }

    # Utiliser le RAG pour analyser
    try:
        # Vue d'ensemble du projet
        overview_results = retriever.retrieve_context_for_readme('project_description')
        analysis['project_overview'] = {
            'chunks_found': len(overview_results),
            'main_files': [r['file_path'] for r in overview_results],
            'key_content': [r['content'][:200] + "..." for r in overview_results[:2]]
        }

        # Informations d'installation
        install_results = retriever.retrieve_context_for_readme('installation')
        analysis['installation_info'] = {
            'dependencies_files': [r['file_path'] for r in install_results if 'dependencies' in r['chunk_type']],
            'setup_files': [r['file_path'] for r in install_results if 'setup' in r['chunk_type']],
            'config_files': [r['file_path'] for r in install_results if 'config' in r['chunk_type']]
        }

        # Exemples d'usage
        usage_results = retriever.retrieve_context_for_readme('usage_examples')
        analysis['usage_examples'] = {
            'entry_points': [r for r in usage_results if 'main' in r['chunk_type']],
            'notebook_examples': [r for r in usage_results if 'notebook' in r['chunk_type']],
            'key_functions': [r for r in usage_results if 'function' in r['chunk_type']]
        }

        # Structure du projet
        structure_results = retriever.retrieve_context_for_readme('project_structure')
        analysis['project_structure'] = {
            'main_classes': [r for r in structure_results if 'class' in r['chunk_type']],
            'modules': [r for r in structure_results if 'module' in r['chunk_type']],
            'total_components': len(structure_results)
        }

    except Exception as e:
        print(f"⚠️ Erreur analyse RAG pour README: {e}")

    return analysis


def print_readme_analysis(analysis):
    """Affiche l'analyse pour README"""
    print(f"\n📝 ANALYSE POUR GÉNÉRATION README:")
    print(f"=" * 40)

    print(f"\n🎯 Vue d'ensemble:")
    overview = analysis['project_overview']
    print(f"   - Chunks trouvés: {overview.get('chunks_found', 0)}")
    print(f"   - Fichiers principaux: {len(overview.get('main_files', []))}")

    print(f"\n📦 Installation:")
    install = analysis['installation_info']
    print(f"   - Fichiers de dépendances: {len(install.get('dependencies_files', []))}")
    print(f"   - Scripts de setup: {len(install.get('setup_files', []))}")
    print(f"   - Fichiers de config: {len(install.get('config_files', []))}")

    print(f"\n🚀 Usage:")
    usage = analysis['usage_examples']
    print(f"   - Points d'entrée: {len(usage.get('entry_points', []))}")
    print(f"   - Notebooks d'exemple: {len(usage.get('notebook_examples', []))}")
    print(f"   - Fonctions clés: {len(usage.get('key_functions', []))}")

    print(f"\n🏗️ Structure:")
    structure = analysis['project_structure']
    print(f"   - Classes principales: {len(structure.get('main_classes', []))}")
    print(f"   - Modules: {len(structure.get('modules', []))}")

    print(f"\n✅ PRÊT POUR GÉNÉRATION README!")


if __name__ == "__main__":
    print("🚀 Lancement du pipeline complet")

    try:
        result = main()

        if result:
            print(f"\n🎉 SUCCÈS! Système RAG complètement opérationnel")
            print(f"💡 Prochaine étape: Agent Orchestrateur pour génération README")
        else:
            print(f"\n❌ Échec du pipeline")

    except KeyboardInterrupt:
        print(f"\n⏹️  Arrêt demandé par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        import traceback

        traceback.print_exc()