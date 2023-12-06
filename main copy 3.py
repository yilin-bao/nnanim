from _ast import Assign, AsyncFunctionDef, Call, ClassDef, FunctionDef
import os
import ast
import importlib.util
from typing import Any

from numpy import isin

def find_python_files(directory):
    python_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files

directory_path = "/Users/zzyang/Documents/NNanim/TestingCode"
python_files = find_python_files(directory_path)

def read_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    

program_stacks = []
program_elements = {}
program_parameters = {}

# ==============================
# Analyze the folder containing
# ==============================

class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.current_class = None
        self.current_function = None
                
    def visit_ClassDef(self, node: ClassDef) -> Any:
        for base in node.bases:
            if isinstance(base, ast.Attribute) and base.attr == "Module":
                self.current_class = node.name
                program_elements[node.name] = {
                    "variables": [],
                    "type": None,
                }
                self.generic_visit(node)
                self.current_class = None

    def visit_FunctionDef(self, node: FunctionDef) -> Any:
        if node.name == '__init__':
            self.current_function = '__init__'
        elif node.name == 'forward':
            self.current_function = 'forward'
        else:
            self.current_function = node.name
            if self.current_class:
                var = []
                for arg in node.args.args:
                    var.append(arg.arg)
                program_elements[f"{self.current_class}.{node.name}"] = {
                    "variables": var,
                    "type": None,
                }
            else:
                program_elements[node.name] = {
                    "variables": [],
                    "type": None,
                }
        self.generic_visit(node)
        self.current_function = None

    def visit_Assign(self, node: Assign) -> Any:
        if self.current_function == '__init__' and self.current_class:
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    program_elements[f"{self.current_class}.{target.attr}"] = {}
        else:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # program_stacks.append((target.id, target.id, target.id))
                    program_stacks.append(target.id)
                        
            

# for python_file in python_files:
#     print("="*65)
#     print("="*5, python_file, "="*5)
#     print("="*65)
#     my_code = read_file_content(python_file)
#     tree = ast.parse(my_code)
#     analyzer = CodeAnalyzer()
#     analyzer.visit(tree)
my_code = read_file_content("/Users/zzyang/Documents/NNanim/TestingCode/modules.py")
# my_code = read_file_content("/Users/zzyang/Documents/NNanim/TestingCode/__init__.py")
tree = ast.parse(my_code)
analyzer = CodeAnalyzer()
analyzer.visit(tree)
for name, data in program_elements.items():
    print(f"Element: {name}")
for pop in program_stacks:
    print(f"Stack: {pop}")