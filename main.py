from _ast import AST, Assign, AsyncFunctionDef, Call, ClassDef, FunctionDef
import os
import ast
import importlib.util
from re import L
from typing import Any
import numpy

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

    def visit_Call(self, node: Call) -> Any:
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Call):
                if isinstance(node.func.value.func, ast.Name):
                    print(node.lineno, f"{node.func.value.func.id}.{node.func.attr}")
            elif isinstance(node.func.value, ast.Name):
                print(node.lineno, f"{node.func.value.id}.{node.func.attr}")
            elif isinstance(node.func.value, ast.Attribute):
                print(node.lineno, f"{node.func.value.attr}.{node.func.attr}")
        self.generic_visit(node)
        


my_code = read_file_content("/Users/zzyang/Documents/NNanim/TestingCode/modules.py")
# my_code = read_file_content("/Users/zzyang/Documents/NNanim/TestingCode/__init__.py")
tree = ast.parse(my_code)
analyzer = CodeAnalyzer()
analyzer.visit(tree)
for name, data in program_elements.items():
    print(f"Element: {name}")
for pop in program_stacks:
    print(f"Stack: {pop}")