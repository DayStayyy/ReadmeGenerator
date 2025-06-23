#!/usr/bin/env python3
"""
Agent Orchestrateur pour génération automatique de README
Utilise le RAG pour poser des questions intelligentes et compiler les réponses
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Import Ollama Python client
try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("❌ ollama package requis. Installe: pip install ollama")
    exit(1)


@dataclass
class ReadmeSection:
    """Représente une section du README"""
    title: str
    content: str
    section_type: str
    generated_from: List[str]  # Sources des chunks utilisés
    confidence: float = 0.0


class ReadmeOrchestrator:
    """Agent orchestrateur pour génération de README"""

    def __init__(self, retriever, project_info: Dict, model_name: str = "llama3.2:latest"):
        """
        retriever: RAGRetriever instance
        project_info: Infos du projet depuis le scanner
        model_name: Nom du modèle Ollama à utiliser
        """
        self.retriever = retriever
        self.project_info = project_info
        self.model_name = model_name
        self.readme_sections = []

        # Vérifier Ollama (obligatoire maintenant)
        self._ensure_ollama_ready()

        print(f"✅ Ollama prêt avec modèle: {model_name}")

        # Template de README structure
        self.readme_template = [
            "project_title",
            "project_description",
            "installation",
            "usage",
            "project_structure",
            "api_reference",
            "examples",
            "contributing",
            "license"
        ]

        # Questions par section
        self.section_questions = {
            "project_title": [
                "What is the main purpose of this project?",
                "What is the project name and main functionality?",
            ],
            "project_description": [
                "What does this project do? What problem does it solve?",
                "What are the main features and capabilities?",
                "What technologies and frameworks are used?",
            ],
            "installation": [
                "What are the project dependencies and requirements?",
                "How to install and setup this project?",
                "What configuration files are needed?",
            ],
            "usage": [
                "What are the main entry points and how to run the project?",
                "What are examples of using the main functions or classes?",
                "How to use the command line interface or main script?",
            ],
            "project_structure": [
                "What are the main modules, classes and components?",
                "How is the project organized and what is the folder structure?",
                "What are the main Python files and their purposes?",
            ],
            "api_reference": [
                "What are the main classes and their methods?",
                "What are the important functions and their parameters?",
                "What are the main APIs and interfaces?",
            ],
            "examples": [
                "Are there any Jupyter notebooks with examples?",
                "What are practical usage examples and code samples?",
                "Are there any demo scripts or example files?",
            ]
        }

    def _ensure_ollama_ready(self):
        """S'assure qu'Ollama est prêt avec le modèle demandé"""

        if not OLLAMA_AVAILABLE:
            print("❌ Module ollama requis")
            print("   Installe: pip install ollama")
            exit(1)

        try:
            # Tester la connexion
            models = ollama.list()
            model_names = [model['model'] for model in models.get('models', [])]

            if self.model_name not in model_names:
                print(f"🔄 Modèle {self.model_name} non trouvé, téléchargement...")

                try:
                    ollama.pull(self.model_name)
                    print(f"✅ Modèle {self.model_name} téléchargé!")
                except Exception as e:
                    print(f"❌ Échec téléchargement modèle: {e}")
                    print("   Démarre Ollama: ollama serve")
                    exit(1)

            # Test rapide du modèle
            test_response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': 'Hello'}]
            )

            if not test_response:
                raise Exception("Pas de réponse du modèle")

        except Exception as e:
            print(f"❌ Erreur Ollama: {e}")
            print("   Solutions:")
            print("   1. Démarre Ollama: ollama serve")
            print("   2. Vérifie le modèle: ollama list")
            print(f"   3. Télécharge le modèle: ollama pull {self.model_name}")
            exit(1)

    def generate_readme(self) -> str:
        """Génère un README complet en orchestrant les questions"""

        print("🤖 Démarrage de l'orchestrateur README")
        print("=" * 50)

        # Générer chaque section
        for section_type in self.readme_template:
            if section_type in ["contributing", "license"]:
                # Sections standard sans RAG
                section = self._generate_standard_section(section_type)
            else:
                # Sections basées sur RAG
                section = self._generate_rag_section(section_type)

            if section:
                self.readme_sections.append(section)
                print(f"✅ Section '{section_type}' générée")
            else:
                print(f"⚠️  Section '{section_type}' ignorée (pas de contenu)")

        # Assembler le README final
        readme_content = self._assemble_readme()

        # Sauvegarder
        self._save_readme(readme_content)

        return readme_content

    def _generate_rag_section(self, section_type: str) -> Optional[ReadmeSection]:
        """Génère une section en utilisant le RAG"""

        questions = self.section_questions.get(section_type, [])
        if not questions:
            return None

        print(f"\n🔍 Génération section: {section_type}")

        # Collecter les informations via RAG
        all_context = []
        sources_used = []

        for question in questions:
            print(f"   Question: {question}")

            # Récupérer le contexte pertinent
            results = self.retriever.retrieve_for_query(question, k=3)

            if results:
                for result in results:
                    all_context.append({
                        'question': question,
                        'content': result['content'],
                        'file_path': result['file_path'],
                        'chunk_type': result['chunk_type'],
                        'score': result['similarity_score']
                    })
                    sources_used.append(result['chunk_id'])

        if not all_context:
            return None

        # Générer le contenu avec LLM
        section_content = self._generate_section_content(section_type, all_context)

        if section_content:
            return ReadmeSection(
                title=self._get_section_title(section_type),
                content=section_content,
                section_type=section_type,
                generated_from=list(set(sources_used)),
                confidence=self._calculate_confidence(all_context)
            )

        return None

    def _generate_section_content(self, section_type: str, context: List[Dict]) -> str:
        """Génère le contenu d'une section avec le LLM"""

        # Préparer le prompt
        prompt = self._create_section_prompt(section_type, context)

        # Appeler le LLM (obligatoire maintenant)
        response = self._call_llm(prompt)

        if response:
            return response.strip()
        else:
            # Si échec LLM, on crash
            raise Exception(f"Échec génération section {section_type} avec LLM")

    def _create_section_prompt(self, section_type: str, context: List[Dict]) -> str:
        """Crée le prompt pour une section"""

    def _create_section_prompt(self, section_type: str, context: List[Dict]) -> str:
        """Crée le prompt pour une section"""

        section_instructions = {
            "project_title": f"Generate ONLY a clear project title (using # header) and brief one-line description for the project named '{self.project_info.get('name', 'Unknown')}'. Do not include any other text or ask for modifications.",
            "project_description": "Write a comprehensive description explaining what this project does, its main features, and technologies used. Start directly with the content, do not include section headers like ## Description.",
            "installation": "Provide clear installation instructions including dependencies and setup steps. Include code blocks for commands. Start directly with the content.",
            "usage": "Show how to use this project with practical examples. Include code examples in markdown code blocks. Start directly with the content.",
            "project_structure": "Describe the project organization, main modules, and key components. Explain the purpose of important files. Start directly with the content.",
            "api_reference": "Document the main classes, functions, and their usage with clear examples. Start directly with the content.",
            "examples": "Provide practical examples and usage scenarios with code snippets. Start directly with the content."
        }

        instruction = section_instructions.get(section_type, "Generate appropriate content for this section.")

        # Construire le contexte plus proprement
        context_parts = []
        for ctx in context[:5]:  # Limiter le contexte
            file_info = f"File: {ctx['file_path']} (Type: {ctx['chunk_type']})"
            content_preview = ctx['content'][:300] + "..." if len(ctx['content']) > 300 else ctx['content']
            context_parts.append(f"{file_info}\n{content_preview}")

        context_text = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are a technical writer creating a professional README section for a {self.project_info.get('project_type', 'Python')} project.

Project Information:
- Name: {self.project_info.get('name', 'Unknown')}
- Type: {self.project_info.get('project_type', 'Python application')}

Task: {instruction}

Code Context:
{context_text}

Requirements:
- Write in clear, professional markdown
- Be concise but informative
- Include relevant code examples where appropriate
- Use proper markdown formatting (code blocks, lists)
- Focus on practical, actionable information
- Do NOT include section headers (like ## Description) - they will be added automatically
- Do NOT ask for modifications or feedback
- Generate ONLY the content for this specific section

Content:"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Appelle le LLM via Ollama Python client - obligatoire"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={
                    'temperature': 0.7,
                    'num_predict': 1500,  # Plus de tokens
                }
            )

            return response['message']['content'].strip()

        except Exception as e:
            print(f"❌ Erreur LLM critique: {e}")
            raise Exception(f"LLM indisponible: {e}")

    def _generate_standard_section(self, section_type: str) -> ReadmeSection:
        """Génère les sections standard (contributing, license)"""

        standard_sections = {
            "contributing": {
                "title": "## Contributing",
                "content": """1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request"""
            },
            "license": {
                "title": "## License",
                "content": "This project is licensed under the MIT License - see the LICENSE file for details."
            }
        }

        if section_type in standard_sections:
            section_data = standard_sections[section_type]
            return ReadmeSection(
                title=section_data["title"],
                content=section_data["content"],
                section_type=section_type,
                generated_from=["template"],
                confidence=1.0
            )

        return None

    def _get_section_title(self, section_type: str) -> str:
        """Retourne le titre formaté pour une section"""
        titles = {
            "project_title": "",  # Pas de titre supplémentaire
            "project_description": "",  # Pas de header pour description
            "installation": "## Installation",
            "usage": "## Usage",
            "project_structure": "## Project Structure",
            "api_reference": "## API Reference",
            "examples": "## Examples"
        }
        return titles.get(section_type, f"## {section_type.replace('_', ' ').title()}")

    def _calculate_confidence(self, context: List[Dict]) -> float:
        """Calcule un score de confiance basé sur le contexte"""
        if not context:
            return 0.0

        scores = [ctx['score'] for ctx in context if 'score' in ctx]
        avg_score = sum(scores) / len(scores) if scores else 0.5

        # Facteur basé sur la quantité de contexte
        quantity_factor = min(len(context) / 5, 1.0)  # Max 1.0 pour 5+ éléments

        return avg_score * quantity_factor

    def _assemble_readme(self) -> str:
        """Assemble le README final"""

        readme_parts = []

        # En-tête avec métadonnées
        readme_parts.append(
            f"<!-- Generated by README Orchestrator on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -->")

        # Ajouter chaque section
        for section in self.readme_sections:
            if section.section_type == "project_title":
                # Titre principal - ne pas ajouter de header supplémentaire
                readme_parts.append(section.content)
            elif section.section_type == "project_description":
                # Description directement après le titre sans header
                readme_parts.append(section.content)
            else:
                # Autres sections avec leur header
                readme_parts.append(f"{section.title}\n\n{section.content}")

        return "\n\n".join(readme_parts)

    def _save_readme(self, content: str):
        """Sauvegarde le README et les métadonnées"""

        # Sauvegarder le README
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(content)

        # Sauvegarder les métadonnées de génération
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "project_info": self.project_info,
            "sections_generated": len(self.readme_sections),
            "sections_details": [
                {
                    "section_type": s.section_type,
                    "title": s.title,
                    "sources_count": len(s.generated_from),
                    "confidence": s.confidence,
                    "sources": s.generated_from
                }
                for s in self.readme_sections
            ]
        }

        with open("readme_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"\n💾 README généré et sauvegardé:")
        print(f"   - README.md")
        print(f"   - readme_metadata.json")

    def interactive_review(self):
        """Permet de réviser interactivement le README généré"""

        print(f"\n📝 RÉVISION INTERACTIVE DU README")
        print(f"Sections générées: {len(self.readme_sections)}")

        for i, section in enumerate(self.readme_sections):
            print(f"\n--- Section {i + 1}: {section.section_type} ---")
            print(f"Confiance: {section.confidence:.2f}")
            print(f"Sources: {len(section.generated_from)}")
            print(f"Titre: {section.title}")
            print(f"Contenu (100 premiers chars): {section.content[:100]}...")

            # Option pour modifier
            choice = input(f"[k]eep, [r]egenerate, [s]kip? (k): ").lower()

            if choice == 'r':
                print("🔄 Régénération de cette section...")
                # TODO: Implémenter régénération avec prompts modifiés
            elif choice == 's':
                print("⏩ Section ignorée")
                self.readme_sections.remove(section)


# Test de l'orchestrateur
if __name__ == "__main__":
    print("🧪 Test de l'orchestrateur README")


    # Simulation pour test
    class MockRetriever:
        def retrieve_for_query(self, query, k=3):
            return [
                {
                    'content': f"Sample code content for query: {query}",
                    'file_path': 'test.py',
                    'chunk_type': 'python_code',
                    'similarity_score': 0.8,
                    'chunk_id': 'test_chunk_1'
                }
            ]


    mock_project_info = {
        'name': 'test_project',
        'project_type': 'Python application',
        'dependencies': {'requirements': ['numpy', 'pandas']}
    }

    # Test avec LLM réel
    orchestrator = ReadmeOrchestrator(
        retriever=MockRetriever(),
        project_info=mock_project_info,
        model_name="llama3.2:latest"
    )

    # Générer une section de test
    test_section = orchestrator._generate_rag_section("installation")

    if test_section:
        print(f"✅ Section de test générée:")
        print(f"   Titre: {test_section.title}")
        print(f"   Type: {test_section.section_type}")
        print(f"   Contenu: {test_section.content[:200]}...")
    else:
        print("❌ Échec génération section de test")