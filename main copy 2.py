from _ast import Call
import os
import ast
import importlib.util
from typing import Any

def find_python_files(directory):
    python_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files

directory_path = "/Users/zzyang/Documents/NNanim/TestingCode"
python_files = find_python_files(directory_path)

program_stack = []
program_element = {}

# ==============================
# 上面是分析一下文件目录
# ==============================

class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.classes = {}
        self.functions = {}
        
        self.current_class = None
        self.current_function = None
        
    def visit_ClassDef(self, node):
        for base in node.bases:
            if isinstance(base, ast.Attribute) and base.attr == "Module":
                self.current_class = node.name
                self.classes[node.name] = {
                    'functions': [],
                }
                self.generic_visit(node)
                self.current_class = None
                break
        
    def visit_FunctionDef(self, node):
        if node.name == '__init__':
            pass
        elif node.name == 'forward':
            pass
        else:
            if self.current_class:
                self.classes[self.current_class]['functions'].append(node.name)
                self.generic_visit(node)
            else:
                self.current_function = node.name
                self.functions[node.name] = {
                    'calls': [],
                }
                self.generic_visit(node)
                self.current_function = None
        
    def visit_Call(self, node: Call) -> Any:
        if self.current_function:
            if isinstance(node.func, ast.Name):
                # print(node.func.id)
                # print(self.current_function)
                self.functions[self.current_function]['calls'].append(node.func.id)
            # if isinstance(node.func, ast.Attribute):
                # self.functions[self.current_function]['calls'].append(node.func.attr)
        self.generic_visit(node)
        
    # def visit_Assign(self, node: Call) -> Any:
    #     for target in node.targets:
    #         if isinstance(target, ast.Name):
    #             variable_name = target.id
    #             value_expression = node.value.func.attr
    #             print(variable_name, value_expression, node.value.args[0].id)


# def analyze_code(code):
#     tree = ast.parse(code)
#     analyzer = CodeAnalyzer()
#     analyzer.visit(tree)
#     return analyzer.classes, analyzer.functions

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

for python_file in python_files:
    my_code = read_file_content(python_file)
    # classes = analyze_code(my_code)
    tree = ast.parse(my_code)
    analyzer = CodeAnalyzer()
    analyzer.visit(tree)
    for class_name, class_data in analyzer.classes.items():
        print(f"Class: {class_name};", f"Functions: {', '.join(class_data['functions'])};")
    for func_name, func_data in analyzer.functions.items():
        print(f"Function: {func_name}")


# @@@@@

for python_file in python_files:
    print("="*65)
    print("="*5, python_file, "="*5)
    print("="*65)
    my_code = read_file_content(python_file)
    tree = ast.parse(my_code)
    analyzer = CodeAnalyzer()
    analyzer.visit(tree)
    # for func_name, func_data in analyzer.functions.items():
    #     print(f"Function: {func_name}", f"Stack: {', '.join(func_data['program_stack'])}")
    
    


class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.functions = {}
        self.current_function = None
        
    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        for base in node.bases:
            if isinstance(base, ast.Attribute) and base.attr == "Module":
                self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.current_function = node.name
        print(node.name)
        # self.functions[node.name] = {}
        self.generic_visit(node)
        self.current_function = None

    def visit_Assign(self, node: Assign) -> Any:
        for target in node.targets:
            # print(target)
            if isinstance(target, ast.Name):
                print("\tVisit Assign Name:", node.lineno, target.id)
                # ast.Call, ast.BinOp, ast.List
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                    print("\tVisit Assign Name (value):", node.lineno, node.value.func.attr)
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                print("\tVisit Assign Attribute:", target.lineno, target.attr)
        self.generic_visit(node)
        
        
        # elif self.current_function == 'forward' and self.current_class:
            # if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            #     if isinstance(node.value.func.value, ast.Name):
            #         if node.value.func.value.id == 'self':
            #             program_stack.append(f"{self.current_class}.{node.value.func.attr}")
            #     elif isinstance(node.value.func.value, ast.Call) and isinstance(node.value.func.value.func, ast.Attribute):
            #         print(node.value.func.value.func)
            #         print(node.value.func.value.func.attr)