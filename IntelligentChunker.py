import ast
import json
import re
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Chunk:
    """Représente un chunk de code/texte avec métadonnées"""
    content: str
    chunk_type: str
    file_path: str
    metadata: Dict[str, Any]
    start_line: int = 0
    end_line: int = 0
    chunk_id: str = ""


class IntelligentChunker:
    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size

        # Mapping handler -> méthode de chunking
        self.chunking_strategies = {
            'python_code': self._chunk_python_code,
            'jupyter_notebook': self._chunk_jupyter_notebook,
            'markdown': self._chunk_markdown,
            'restructured_text': self._chunk_restructured_text,
            'yaml_config': self._chunk_config_file,
            'json_config': self._chunk_config_file,
            'toml_config': self._chunk_config_file,
            'requirements': self._chunk_requirements,
            'setup_script': self._chunk_setup_script,
            'generic_text': self._chunk_generic_text,
            'text_document': self._chunk_text_document
        }

    def chunk_all_files(self, files_data: List[Dict]) -> List[Chunk]:
        """Chunk tous les fichiers selon leur type"""
        all_chunks = []

        for file_data in files_data:
            try:
                chunks = self.chunk_file(file_data)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Erreur chunking {file_data['path']}: {e}")
                # Fallback: chunk générique
                chunks = self._chunk_generic_text(file_data)
                all_chunks.extend(chunks)

        # Ajouter des IDs uniques
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_id = f"chunk_{i:04d}"

        return all_chunks

    def chunk_file(self, file_data: Dict) -> List[Chunk]:
        """Chunk un fichier selon son handler"""
        handler = file_data.get('handler', 'generic_text')
        chunking_method = self.chunking_strategies.get(handler, self._chunk_generic_text)

        return chunking_method(file_data)

    def _chunk_python_code(self, file_data: Dict) -> List[Chunk]:
        """Chunking spécialisé pour code Python"""
        chunks = []
        content = file_data['content']
        file_path = file_data['path']
        metadata = file_data.get('metadata', {})

        try:
            tree = ast.parse(content)
            lines = content.splitlines()

            # Chunk 1: Imports et docstring du module
            imports_and_docstring = []
            module_docstring = ast.get_docstring(tree)

            if module_docstring:
                imports_and_docstring.append(f'"""Module docstring:\n{module_docstring}\n"""')

            # Collecter tous les imports
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                    imports_and_docstring.append(import_line)

            if imports_and_docstring:
                chunks.append(Chunk(
                    content="\n".join(imports_and_docstring),
                    chunk_type="module_header",
                    file_path=file_path,
                    metadata={"imports": metadata.get('imports', [])},
                    start_line=1,
                    end_line=len(imports_and_docstring)
                ))

            # Chunk 2-N: Une chunk par classe
            for class_info in metadata.get('classes', []):
                class_node = self._find_class_node(tree, class_info['name'])
                if class_node:
                    class_content = self._extract_node_content(class_node, lines)

                    chunks.append(Chunk(
                        content=class_content,
                        chunk_type="class_definition",
                        file_path=file_path,
                        metadata={
                            "class_name": class_info['name'],
                            "docstring": class_info.get('docstring'),
                            "methods": self._get_class_methods(class_node)
                        },
                        start_line=class_node.lineno,
                        end_line=class_node.end_lineno or class_node.lineno
                    ))

            # Chunk pour les fonctions standalone
            standalone_functions = []
            for func_info in metadata.get('functions', []):
                # Vérifier si la fonction n'est pas dans une classe
                if not self._is_function_in_class(tree, func_info['name']):
                    func_node = self._find_function_node(tree, func_info['name'])
                    if func_node:
                        func_content = self._extract_node_content(func_node, lines)
                        standalone_functions.append(func_content)

            if standalone_functions:
                chunks.append(Chunk(
                    content="\n\n".join(standalone_functions),
                    chunk_type="standalone_functions",
                    file_path=file_path,
                    metadata={"function_count": len(standalone_functions)},
                    start_line=0,
                    end_line=len(lines)
                ))

            # Chunk pour le code principal (if __name__ == "__main__")
            if metadata.get('has_if_main'):
                main_content = self._extract_main_block(content)
                if main_content:
                    chunks.append(Chunk(
                        content=main_content,
                        chunk_type="main_execution",
                        file_path=file_path,
                        metadata={"is_entry_point": True},
                        start_line=0,
                        end_line=len(lines)
                    ))

        except Exception as e:
            # Fallback: chunking par taille
            return self._chunk_by_size(content, file_path, "python_fallback")

        return chunks

    def _chunk_jupyter_notebook(self, file_data: Dict) -> List[Chunk]:
        """Chunking spécialisé pour notebooks Jupyter"""
        chunks = []
        metadata = file_data.get('metadata', {})
        file_path = file_data['path']

        # Chunk pour chaque cellule de code
        for i, code_cell in enumerate(metadata.get('code_cells', [])):
            if code_cell.strip():
                chunks.append(Chunk(
                    content=code_cell,
                    chunk_type="notebook_code_cell",
                    file_path=file_path,
                    metadata={"cell_number": i + 1},
                    start_line=0,
                    end_line=len(code_cell.splitlines())
                ))

        # Chunk pour les cellules markdown (documentation)
        markdown_content = "\n\n".join(metadata.get('markdown_cells', []))
        if markdown_content.strip():
            chunks.append(Chunk(
                content=markdown_content,
                chunk_type="notebook_documentation",
                file_path=file_path,
                metadata={"cell_count": len(metadata.get('markdown_cells', []))},
                start_line=0,
                end_line=len(markdown_content.splitlines())
            ))

        return chunks

    def _chunk_markdown(self, file_data: Dict) -> List[Chunk]:
        """Chunking spécialisé pour Markdown"""
        chunks = []
        content = file_data['content']
        file_path = file_data['path']

        # Séparer par sections (headers #, ##, ###)
        sections = re.split(r'\n(?=#{1,6}\s)', content)

        for i, section in enumerate(sections):
            if section.strip():
                # Détecter le niveau de header
                header_match = re.match(r'^(#{1,6})\s+(.+)', section)
                header_level = len(header_match.group(1)) if header_match else 0
                header_title = header_match.group(2) if header_match else f"Section {i + 1}"

                chunks.append(Chunk(
                    content=section.strip(),
                    chunk_type="markdown_section",
                    file_path=file_path,
                    metadata={
                        "header_level": header_level,
                        "header_title": header_title,
                        "section_number": i + 1
                    },
                    start_line=0,
                    end_line=len(section.splitlines())
                ))

        return chunks

    def _chunk_restructured_text(self, file_data: Dict) -> List[Chunk]:
        """Chunking spécialisé pour ReStructured Text"""
        chunks = []
        content = file_data['content']
        file_path = file_data['path']

        # Séparer par sections (headers avec === ou --- sous le titre)
        sections = re.split(r'\n(?=.+\n[=\-~`#"^+*:\'<>_]{3,})', content)

        for i, section in enumerate(sections):
            if section.strip():
                # Détecter le titre de section
                lines = section.strip().split('\n')
                title = lines[0] if len(lines) > 1 else f"Section {i + 1}"

                chunks.append(Chunk(
                    content=section.strip(),
                    chunk_type="rst_section",
                    file_path=file_path,
                    metadata={
                        "section_title": title,
                        "section_number": i + 1
                    },
                    start_line=0,
                    end_line=len(section.splitlines())
                ))

        # Si pas de sections détectées, traiter comme un seul chunk
        if not chunks:
            chunks.append(Chunk(
                content=content,
                chunk_type="rst_document",
                file_path=file_path,
                metadata={"is_single_document": True},
                start_line=1,
                end_line=len(content.splitlines())
            ))

        return chunks

    def _chunk_config_file(self, file_data: Dict) -> List[Chunk]:
        """Chunking pour fichiers de configuration"""
        content = file_data['content']
        file_path = file_data['path']

        # Garder les configs en un seul chunk (généralement petites)
        return [Chunk(
            content=content,
            chunk_type="configuration",
            file_path=file_path,
            metadata={
                "config_type": file_data.get('extension', ''),
                "is_project_config": True
            },
            start_line=1,
            end_line=len(content.splitlines())
        )]

    def _chunk_requirements(self, file_data: Dict) -> List[Chunk]:
        """Chunking pour requirements.txt"""
        content = file_data['content']
        file_path = file_data['path']

        # Parser les dépendances
        lines = content.splitlines()
        dependencies = []
        comments = []

        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                comments.append(line)
            elif line and not line.startswith('-'):
                dependencies.append(line)

        return [Chunk(
            content=content,
            chunk_type="project_dependencies",
            file_path=file_path,
            metadata={
                "dependency_count": len(dependencies),
                "dependencies": dependencies,
                "has_comments": len(comments) > 0
            },
            start_line=1,
            end_line=len(lines)
        )]

    def _chunk_setup_script(self, file_data: Dict) -> List[Chunk]:
        """Chunking pour setup.py"""
        # Traiter comme du code Python mais avec métadonnées spéciales
        chunks = self._chunk_python_code(file_data)

        # Modifier le type pour indiquer que c'est un setup script
        for chunk in chunks:
            chunk.chunk_type = f"setup_{chunk.chunk_type}"
            chunk.metadata["is_setup_script"] = True

        return chunks

    def _chunk_generic_text(self, file_data: Dict) -> List[Chunk]:
        """Chunking générique par taille"""
        content = file_data['content']
        file_path = file_data['path']

        return self._chunk_by_size(content, file_path, "generic_text")

    def _chunk_text_document(self, file_data: Dict) -> List[Chunk]:
        """Chunking pour documents texte"""
        content = file_data['content']
        file_path = file_data['path']

        # Si c'est un README, traiter spécialement
        if 'readme' in file_path.lower():
            return [Chunk(
                content=content,
                chunk_type="readme_documentation",
                file_path=file_path,
                metadata={"is_main_documentation": True},
                start_line=1,
                end_line=len(content.splitlines())
            )]

        return self._chunk_by_size(content, file_path, "text_document")

    # Méthodes utilitaires
    def _chunk_by_size(self, content: str, file_path: str, chunk_type: str) -> List[Chunk]:
        """Chunking basique par taille"""
        chunks = []
        lines = content.splitlines()

        current_chunk = []
        current_size = 0
        start_line = 1

        for i, line in enumerate(lines, 1):
            current_chunk.append(line)
            current_size += len(line)

            if current_size >= self.max_chunk_size:
                chunks.append(Chunk(
                    content="\n".join(current_chunk),
                    chunk_type=chunk_type,
                    file_path=file_path,
                    metadata={"line_count": len(current_chunk)},
                    start_line=start_line,
                    end_line=i
                ))

                current_chunk = []
                current_size = 0
                start_line = i + 1

        # Dernier chunk
        if current_chunk:
            chunks.append(Chunk(
                content="\n".join(current_chunk),
                chunk_type=chunk_type,
                file_path=file_path,
                metadata={"line_count": len(current_chunk)},
                start_line=start_line,
                end_line=len(lines)
            ))

        return chunks

    def _find_class_node(self, tree: ast.AST, class_name: str) -> ast.ClassDef:
        """Trouve un nœud de classe par nom"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return node
        return None

    def _find_function_node(self, tree: ast.AST, func_name: str) -> ast.FunctionDef:
        """Trouve un nœud de fonction par nom"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return node
        return None

    def _extract_node_content(self, node: ast.AST, lines: List[str]) -> str:
        """Extrait le contenu textuel d'un nœud AST"""
        start_line = node.lineno - 1
        end_line = (node.end_lineno or node.lineno) - 1

        return "\n".join(lines[start_line:end_line + 1])

    def _get_class_methods(self, class_node: ast.ClassDef) -> List[str]:
        """Récupère les noms des méthodes d'une classe"""
        methods = []
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                methods.append(node.name)
        return methods

    def _is_function_in_class(self, tree: ast.AST, func_name: str) -> bool:
        """Vérifie si une fonction est définie dans une classe"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name == func_name:
                        return True
        return False

    def _extract_main_block(self, content: str) -> str:
        """Extrait le bloc if __name__ == "__main__" """
        lines = content.splitlines()
        main_lines = []
        in_main_block = False

        for line in lines:
            if 'if __name__ == "__main__"' in line:
                in_main_block = True
                main_lines.append(line)
            elif in_main_block:
                if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    # Fin du bloc main
                    break
                main_lines.append(line)

        return "\n".join(main_lines) if main_lines else ""


# Test du chunker
if __name__ == "__main__":
    # Exemple de test avec des données mockées
    chunker = IntelligentChunker(max_chunk_size=800)

    # Simulation d'un fichier Python
    test_file = {
        'path': 'example.py',
        'handler': 'python_code',
        'content': '''"""Module de test pour le chunker"""
import os
import sys

class TestClass:
    """Une classe de test"""

    def __init__(self):
        self.value = 42

    def method1(self):
        """Première méthode"""
        return self.value

def standalone_function():
    """Fonction standalone"""
    return "Hello"

if __name__ == "__main__":
    test = TestClass()
    print(test.method1())
''',
        'metadata': {
            'classes': [{'name': 'TestClass', 'docstring': 'Une classe de test'}],
            'functions': [{'name': '__init__'}, {'name': 'method1'}, {'name': 'standalone_function'}],
            'imports': ['os', 'sys'],
            'has_if_main': True
        }
    }

    chunks = chunker.chunk_file(test_file)

    print(f"=== CHUNKING RESULTS ===")
    print(f"Total chunks: {len(chunks)}")

    for chunk in chunks:
        print(f"\nChunk Type: {chunk.chunk_type}")
        print(f"File: {chunk.file_path}")
        print(f"Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"Metadata: {chunk.metadata}")
        print(f"Content preview: {chunk.content[:100]}...")