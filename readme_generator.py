#!/usr/bin/env python3
"""
Générateur automatique de README avec RAG
Usage: python readme_generator.py --folder /path/to/project
"""

import argparse
import sys
import os
from pathlib import Path
import time
from datetime import datetime

# Imports des modules locaux
try:
    from ProjectScanner import ProjectScanner
    from IntelligentChunker import IntelligentChunker
    from EmbeddingVectordb import CodeEmbedder, VectorDatabase, RAGRetriever
    from ReadmeOrchestrator import ReadmeOrchestrator
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("\nFichiers requis dans le même dossier:")
    print("- ProjectScanner.py")
    print("- IntelligentChunker.py")
    print("- embedding_vectordb.py")
    print("- readme_orchestrator.py")
    sys.exit(1)


def parse_arguments():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(
        description="Générateur automatique de README avec RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python readme_generator.py --folder /path/to/project
  python readme_generator.py --folder . --model codellama:7b
  python readme_generator.py --folder ../mon_projet --output custom_readme.md
        """
    )

    parser.add_argument(
        "--folder",
        required=True,
        help="Chemin vers le dossier du projet à analyser"
    )

    parser.add_argument(
        "--model",
        default="llama3.2:latest",
        help="Modèle Ollama à utiliser (défaut: llama3.2:latest)"
    )

    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Modèle d'embedding (défaut: all-MiniLM-L6-v2)"
    )

    parser.add_argument(
        "--output",
        default="README.md",
        help="Nom du fichier README de sortie (défaut: README.md)"
    )

    parser.add_argument(
        "--rag-db",
        default="auto",
        help="Nom de la base RAG (défaut: auto-généré)"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Taille max des chunks (défaut: 1000)"
    )

    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Mode sans LLM (utilise seulement le RAG pour extraction)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mode verbeux"
    )

    return parser.parse_args()


def validate_arguments(args):
    """Valide les arguments"""

    # Vérifier que le dossier existe
    project_path = Path(args.folder).resolve()
    if not project_path.exists():
        print(f"❌ Erreur: Le dossier '{args.folder}' n'existe pas")
        sys.exit(1)

    if not project_path.is_dir():
        print(f"❌ Erreur: '{args.folder}' n'est pas un dossier")
        sys.exit(1)

    args.folder = str(project_path)

    # Générer nom base RAG si auto
    if args.rag_db == "auto":
        safe_name = project_path.name.replace(" ", "_").replace("-", "_")
        args.rag_db = f"{safe_name}_rag_db"

    return args


def print_header():
    """Affiche l'en-tête du script"""
    print("🤖 GÉNÉRATEUR AUTOMATIQUE DE README")
    print("=" * 50)
    print(f"🕒 Démarré à: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def print_step(step_num: int, total: int, title: str):
    """Affiche le numéro d'étape"""
    print(f"\n📍 ÉTAPE {step_num}/{total}: {title}")
    print("-" * 40)


def main():
    """Fonction principale"""

    # Parser et valider arguments
    args = parse_arguments()
    args = validate_arguments(args)

    print_header()

    if args.verbose:
        print(f"📁 Projet: {args.folder}")
        print(f"🤖 Modèle LLM: {args.model}")
        print(f"🧠 Modèle embedding: {args.embedding_model}")
        print(f"📄 Sortie: {args.output}")
        print(f"🗄️  Base RAG: {args.rag_db}")

    start_time = time.time()

    try:
        # ÉTAPE 1: SCANNING
        print_step(1, 6, "Analyse du projet")

        scanner = ProjectScanner()
        scan_result = scanner.scan_project(args.folder)

        print(f"✅ Projet analysé:")
        print(f"   - Type: {scan_result['project_type']}")
        print(f"   - Fichiers: {len(scan_result['all_files'])}")

        if args.verbose:
            handlers_count = {}
            for file_data in scan_result['all_files']:
                handler = file_data['handler']
                handlers_count[handler] = handlers_count.get(handler, 0) + 1

            print(f"   - Types de fichiers:")
            for handler, count in sorted(handlers_count.items()):
                print(f"     * {handler}: {count}")

        # ÉTAPE 2: CHUNKING
        print_step(2, 6, "Découpage intelligent")

        chunker = IntelligentChunker(max_chunk_size=args.chunk_size)
        all_chunks = chunker.chunk_all_files(scan_result['all_files'])

        print(f"✅ Chunking terminé:")
        print(f"   - Total chunks: {len(all_chunks)}")

        if args.verbose:
            chunk_types = {}
            for chunk in all_chunks:
                chunk_type = chunk.chunk_type
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

            print(f"   - Types de chunks:")
            for chunk_type, count in sorted(chunk_types.items()):
                print(f"     * {chunk_type}: {count}")

        # ÉTAPE 3: EMBEDDING
        print_step(3, 6, "Création des embeddings")

        embedder = CodeEmbedder(args.embedding_model)

        if not embedder.model:
            print("❌ Impossible de charger le modèle d'embedding")
            print("   Installe: pip install sentence-transformers")
            sys.exit(1)

        embedded_chunks = embedder.embed_chunks(all_chunks)

        if not embedded_chunks:
            print("❌ Échec création des embeddings")
            sys.exit(1)

        print(f"✅ Embeddings créés:")
        print(f"   - Dimension: {embedder.embedding_dim}")
        print(f"   - Chunks vectorisés: {len(embedded_chunks)}")

        # ÉTAPE 4: BASE VECTORIELLE
        print_step(4, 6, "Construction base RAG")

        vector_db = VectorDatabase()
        vector_db.build_index(embedded_chunks)

        if not vector_db.index:
            print("❌ Échec construction de l'index")
            print("   Installe: pip install faiss-cpu")
            sys.exit(1)

        # Sauvegarder la base
        vector_db.save(args.rag_db)

        print(f"✅ Base RAG construite:")
        print(f"   - Index: {vector_db.index.ntotal} vecteurs")
        print(f"   - Sauvegardé: {args.rag_db}.faiss")

        # ÉTAPE 5: INITIALISATION RAG
        print_step(5, 6, "Initialisation du RAG")

        retriever = RAGRetriever(embedder, vector_db)

        # Test rapide
        test_results = retriever.retrieve_for_query("main purpose of this project", k=2)
        print(f"✅ RAG opérationnel:")
        print(f"   - Test: {len(test_results)} résultats trouvés")

        # ÉTAPE 6: GÉNÉRATION README
        print_step(6, 6, "Génération du README")

        orchestrator = ReadmeOrchestrator(
            retriever=retriever,
            project_info=scan_result['project_info'],
            model_name=args.model
        )

        readme_content = orchestrator.generate_readme()

        # Créer le dossier generations s'il n'existe pas
        generations_dir = Path("generations")
        generations_dir.mkdir(exist_ok=True)

        # Sauvegarder avec nom personnalisé
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(readme_content)

        # Sauvegarder aussi dans le dossier generations avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = Path(args.folder).name
        generations_filename = f"README_{project_name}_{timestamp}.md"
        generations_path = generations_dir / generations_filename

        with open(generations_path, "w", encoding="utf-8") as f:
            # Ajouter header avec infos de génération
            header = f"""<!-- README généré automatiquement -->
<!-- Projet: {project_name} -->
<!-- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -->
<!-- Modèle LLM: {args.model} -->
<!-- Modèle Embedding: {args.embedding_model} -->
<!-- Chunks traités: {len(all_chunks)} -->
<!-- Mode: LLM -->

"""
            f.write(header + readme_content)

        print(f"✅ README généré:")
        print(f"   - Fichier principal: {output_path.resolve()}")
        print(f"   - Archivé dans: {generations_path}")
        print(f"   - Sections: {len(orchestrator.readme_sections)}")

        # RÉSUMÉ FINAL
        elapsed_time = time.time() - start_time

        print(f"\n🎉 GÉNÉRATION TERMINÉE!")
        print("=" * 50)
        print(f"⏱️  Temps total: {elapsed_time:.1f}s")
        print(f"📁 Projet analysé: {scan_result['project_info']['name']}")
        print(f"📄 README généré: {output_path.resolve()}")
        print(f"📁 Archivé dans: {generations_path}")
        print(f"🗄️  Base RAG: {args.rag_db}.faiss")

        print(f"🤖 Modèle utilisé: {args.model}")

        # Afficher aperçu du README
        print(f"\n📖 APERÇU DU README:")
        print("-" * 30)
        preview_lines = readme_content.split('\n')[:10]
        for line in preview_lines:
            print(f"  {line}")
        if len(readme_content.split('\n')) > 10:
            print("  ...")

        print(f"\n✨ README prêt à utiliser!")

    except KeyboardInterrupt:
        print(f"\n⏹️  Arrêt demandé par l'utilisateur")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()