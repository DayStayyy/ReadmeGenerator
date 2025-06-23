import os
import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json


class ProjectScanner:
    def __init__(self):
        # SEULEMENT les patterns à ignorer (approche inclusive)
        self.ignore_patterns = {
            '__pycache__', '.git', '.venv', 'venv', 'env',
            'node_modules', '.pytest_cache', '.mypy_cache',
            'dist', 'build', '*.egg-info', '.tox', '.coverage',
            '.DS_Store', 'Thumbs.db'
        }

        # Extensions binaires à ignorer
        self.binary_extensions = {
            '.exe', '.dll', '.so', '.dylib', '.bin', '.dat',
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico',
            '.mp3', '.mp4', '.avi', '.mov', '.zip', '.tar', '.gz'
        }

        # Fichiers avec traitement spécial
        self.special_handlers = {
            '.py': 'python_code',
            '.ipynb': 'jupyter_notebook',
            '.md': 'markdown',
            '.rst': 'restructured_text',
            '.txt': 'plain_text',
            '.yml': 'yaml_config',
            '.yaml': 'yaml_config',
            '.json': 'json_config',
            '.toml': 'toml_config',
            '.cfg': 'ini_config',
            '.ini': 'ini_config',
            '.sh': 'shell_script',
            '.bat': 'batch_script',
            '.dockerfile': 'dockerfile',
            'dockerfile': 'dockerfile',
            'makefile': 'makefile',
            'requirements.txt': 'requirements',
            'setup.py': 'setup_script',
            'pyproject.toml': 'project_config'
        }

    def scan_project(self, project_path: str) -> Dict:
        """Scan complet du projet - TOUT sauf ce qui est ignoré"""
        project_path = Path(project_path)

        result = {
            'project_info': self._analyze_project_structure(project_path),
            'all_files': self._get_all_files(project_path),  # TOUS les fichiers
            'dependencies': self._extract_dependencies(project_path),
            'project_type': self._detect_project_type(project_path)
        }

        return result

    def _should_ignore(self, path: Path) -> bool:
        """Détermine si un fichier/dossier doit être ignoré"""
        for ignore in self.ignore_patterns:
            if ignore in str(path) or path.name.startswith('.'):
                return True
        return False

    def _should_ignore(self, path: Path) -> bool:
        """Détermine si un fichier/dossier doit être ignoré"""
        # Ignorer les dossiers/fichiers cachés
        if path.name.startswith('.') and path.name not in {'.gitignore', '.env.example'}:
            return True

        # Ignorer selon patterns
        for ignore in self.ignore_patterns:
            if ignore in str(path):
                return True

        # Ignorer les binaires
        if path.suffix.lower() in self.binary_extensions:
            return True

        # Ignorer les fichiers trop gros (>10MB)
        try:
            if path.is_file() and path.stat().st_size > 10 * 1024 * 1024:
                return True
        except:
            pass

        return False

    def _get_file_handler(self, file_path: Path) -> str:
        """Détermine le handler approprié pour un fichier"""
        # Vérifier le nom exact du fichier d'abord
        if file_path.name.lower() in self.special_handlers:
            return self.special_handlers[file_path.name.lower()]

        # Puis l'extension
        if file_path.suffix.lower() in self.special_handlers:
            return self.special_handlers[file_path.suffix.lower()]

        # Par défaut
        return 'generic_text'

    def _get_all_files(self, project_path: Path) -> List[Dict]:
        """Récupère TOUS les fichiers avec leur handler approprié"""
        all_files = []

        for file_path in project_path.rglob("*"):
            if not file_path.is_file() or self._should_ignore(file_path):
                continue

            try:
                handler_type = self._get_file_handler(file_path)
                relative_path = str(file_path.relative_to(project_path))

                # Traitement selon le type
                file_data = {
                    'path': relative_path,
                    'handler': handler_type,
                    'size': file_path.stat().st_size,
                    'extension': file_path.suffix.lower()
                }

                # Ajouter le contenu selon le handler
                content_data = self._process_file_content(file_path, handler_type)
                file_data.update(content_data)

                all_files.append(file_data)

            except Exception as e:
                print(f"Erreur traitement {file_path}: {e}")
                continue

        return all_files

    def _process_file_content(self, file_path: Path, handler_type: str) -> Dict:
        """Traite le contenu selon le type de handler"""
        try:
            if handler_type == 'python_code':
                return self._process_python_file(file_path)

            elif handler_type == 'jupyter_notebook':
                return self._process_jupyter_notebook(file_path)

            elif handler_type in ['markdown', 'restructured_text', 'plain_text']:
                return self._process_text_file(file_path)

            elif handler_type in ['yaml_config', 'json_config', 'toml_config', 'ini_config']:
                return self._process_config_file(file_path)

            elif handler_type in ['requirements', 'setup_script', 'project_config']:
                return self._process_special_file(file_path)

            elif handler_type == 'generic_text':
                return self._process_generic_file(file_path)

            else:
                return {'content': 'Non traité', 'metadata': {}}

        except Exception as e:
            return {'content': f'Erreur: {str(e)}', 'metadata': {'error': True}}

    def _process_python_file(self, file_path: Path) -> Dict:
        """Traite un fichier Python avec analyse AST complète"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        metadata = self._analyze_python_file(content, file_path)
        return {
            'content': content,
            'metadata': metadata,
            'content_type': 'python_source'
        }

    def _process_jupyter_notebook(self, file_path: Path) -> Dict:
        """Traite un notebook Jupyter"""
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook_json = json.load(f)

        # Extraire le code et markdown des cellules
        code_cells = []
        markdown_cells = []

        for cell in notebook_json.get('cells', []):
            if cell.get('cell_type') == 'code':
                code_content = ''.join(cell.get('source', []))
                if code_content.strip():
                    code_cells.append(code_content)

            elif cell.get('cell_type') == 'markdown':
                md_content = ''.join(cell.get('source', []))
                if md_content.strip():
                    markdown_cells.append(md_content)

        return {
            'content': json.dumps(notebook_json, indent=2),
            'metadata': {
                'code_cells': code_cells,
                'markdown_cells': markdown_cells,
                'total_cells': len(notebook_json.get('cells', [])),
                'notebook_type': 'jupyter'
            },
            'content_type': 'jupyter_notebook'
        }

    def _process_text_file(self, file_path: Path) -> Dict:
        """Traite les fichiers texte (MD, RST, TXT)"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return {
            'content': content,
            'metadata': {
                'line_count': len(content.splitlines()),
                'is_documentation': file_path.name.lower().startswith('readme')
            },
            'content_type': 'text_document'
        }

    def _process_config_file(self, file_path: Path) -> Dict:
        """Traite les fichiers de configuration"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return {
            'content': content,
            'metadata': {
                'config_type': file_path.suffix,
                'is_project_config': True
            },
            'content_type': 'configuration'
        }

    def _process_special_file(self, file_path: Path) -> Dict:
        """Traite les fichiers spéciaux (requirements, setup, etc.)"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return {
            'content': content,
            'metadata': {
                'special_type': file_path.name,
                'is_critical': True
            },
            'content_type': 'project_definition'
        }

    def _process_generic_file(self, file_path: Path) -> Dict:
        """Traite les fichiers génériques (essaie de lire comme texte)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {
                'content': content,
                'metadata': {'readable_as_text': True},
                'content_type': 'generic_text'
            }
        except:
            return {
                'content': 'Fichier binaire ou non lisible',
                'metadata': {'readable_as_text': False},
                'content_type': 'binary_or_unreadable'
            }

    def _analyze_python_file(self, content: str, file_path: Path) -> Dict:
        """Analyse AST d'un fichier Python"""
        try:
            tree = ast.parse(content)

            metadata = {
                'classes': [],
                'functions': [],
                'imports': [],
                'docstring': ast.get_docstring(tree),
                'is_main': '__main__' in content,
                'has_if_main': 'if __name__ == "__main__"' in content
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    metadata['classes'].append({
                        'name': node.name,
                        'line': node.lineno,
                        'docstring': ast.get_docstring(node)
                    })

                elif isinstance(node, ast.FunctionDef):
                    metadata['functions'].append({
                        'name': node.name,
                        'line': node.lineno,
                        'docstring': ast.get_docstring(node),
                        'is_private': node.name.startswith('_')
                    })

                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        metadata['imports'].append(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        metadata['imports'].append(node.module)

            return metadata

        except Exception as e:
            return {'error': str(e)}

    def _extract_dependencies(self, project_path: Path) -> Dict:
        """Extrait les dépendances du projet"""
        deps = {
            'requirements': [],
            'setup_requires': [],
            'dev_requires': []
        }

        # requirements.txt
        req_file = project_path / "requirements.txt"
        if req_file.exists():
            try:
                with open(req_file, 'r') as f:
                    deps['requirements'] = [
                        line.strip() for line in f
                        if line.strip() and not line.startswith('#')
                    ]
            except:
                pass

        # setup.py
        setup_file = project_path / "setup.py"
        if setup_file.exists():
            try:
                with open(setup_file, 'r') as f:
                    setup_content = f.read()
                    # Regex simple pour extraire install_requires
                    import re
                    matches = re.findall(r'install_requires\s*=\s*\[(.*?)\]', setup_content, re.DOTALL)
                    if matches:
                        deps['setup_requires'] = [
                            dep.strip().strip('\'"')
                            for dep in matches[0].split(',')
                            if dep.strip()
                        ]
            except:
                pass

        return deps

    def _detect_project_type(self, project_path: Path) -> str:
        """Détecte le type de projet Python"""

        # Vérifications dans l'ordre de spécificité
        if (project_path / "manage.py").exists():
            return "django"

        if (project_path / "app.py").exists() or (project_path / "main.py").exists():
            # Vérifie si c'est Flask
            for py_file in project_path.rglob("*.py"):
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                        if 'from flask import' in content or 'import flask' in content:
                            return "flask"
                        if 'from fastapi import' in content or 'import fastapi' in content:
                            return "fastapi"
                except:
                    continue

        if (project_path / "setup.py").exists() or (project_path / "pyproject.toml").exists():
            return "package"

        if (project_path / "requirements.txt").exists():
            return "application"

        return "script_collection"

    def _analyze_project_structure(self, project_path: Path) -> Dict:
        """Analyse la structure générale du projet"""
        structure = {
            'name': project_path.name,
            'total_files': 0,
            'python_files_count': 0,
            'directories': [],
            'main_modules': []
        }

        # Compter les fichiers
        for item in project_path.rglob("*"):
            if item.is_file() and not self._should_ignore(item):
                structure['total_files'] += 1
                if item.suffix == '.py':
                    structure['python_files_count'] += 1

        # Identifier les dossiers principaux
        for item in project_path.iterdir():
            if item.is_dir() and not self._should_ignore(item):
                structure['directories'].append(item.name)

        # Identifier les modules principaux (dossiers avec __init__.py)
        for py_file in project_path.rglob("__init__.py"):
            if not self._should_ignore(py_file):
                module_path = py_file.parent.relative_to(project_path)
                structure['main_modules'].append(str(module_path))

        return structure


# Test du scanner
if __name__ == "__main__":
    scanner = ProjectScanner()

    # Test sur le projet courant
    project_path = "../VoitureGoBrrrr"  # Remplace par ton chemin
    result = scanner.scan_project(project_path)

    print("=== ANALYSE DU PROJET ===")
    print(f"Type: {result['project_type']}")
    print(f"Total fichiers: {len(result['all_files'])}")
    print(f"Dépendances: {len(result['dependencies']['requirements'])}")

    # Résumé par type de handler
    handlers = {}
    for file_data in result['all_files']:
        handler = file_data['handler']
        handlers[handler] = handlers.get(handler, 0) + 1

    print(f"\n=== TYPES DE FICHIERS ===")
    for handler, count in handlers.items():
        print(f"{handler}: {count} fichiers")

    print("\n=== STRUCTURE ===")
    print(json.dumps(result['project_info'], indent=2))