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

# print("Python Files:")
# for python_file in python_files:
#     print(python_file)

# def get_module_file_path(module_name):
#     try:
#         spec = importlib.util.find_spec(module_name)
#         if spec:
#             return spec.origin
#     except Exception as e:
#         print(f"Error finding file path for {module_name}: {e}")
#     return None

class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.classes = {}
        self.functions = {}
        self.ext_refs = {}
        
        self.current_class = None
        self.current_attr = None

    def visit_ClassDef(self, node):
        self.current_class = node.name
        self.classes[node.name] = {
            'functions': [],
            'variables': [],
            'nn_modules': []
        }
        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node):
        if self.current_class:
            if node.name == '__init__':
                for arg in node.args.args:
                    self.classes[self.current_class]['variables'].append(arg.arg)
            self.classes[self.current_class]['functions'].append(node.name)
        else:
            self.functions[node.name] = {
                'variables': []
            }
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.ext_refs[alias.name] = {}
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module_name = node.module
        for alias in node.names:
            self.ext_refs[f"{module_name}.{alias.name}"] = {}
        self.generic_visit(node)
        
    def visit_Call(self, node: Call) -> Any:
        # print("Visit Call", node.func)
        if isinstance(node.func, ast.Name):
            print("Visit Name:", node.lineno, node.func.id)
            # for arg in node.args:
            #     if isinstance(arg, ast.Name):
            #         print(arg.id)
        if isinstance(node.func, ast.Attribute):
            print("Visit Attribute:", node.lineno, node.func.attr)
        self.generic_visit(node)

    # def visit_Name(self, node):
    #     if self.current_attr:
    #         print("Visit name:", node.lineno, node.id)
    #     self.generic_visit(node)
    
    # def visit_Attribute(self, node):
    #     print("Visit Attribute:", node.lineno, node.attr, node.value)
        # if isinstance(node.value, ast.Name):
        #     print(node.value.id)
        
    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                variable_name = target.id
                value_expression = node.value
                print(variable_name, value_expression)
        self.generic_visit(node)
        
    # def visit_Assign(self, node):
        # print("Visit Assign:", node.lineno, node.targets)
    #     count = 0
    #   for target in node.targets:
    #         # if isinstance(target, ast.Name):
    #         #     print("Visit Assign Name:", node.lineno, target.id)
    #         if isinstance(target, ast.Attribute):
    #             self.current_attr = target.attr
    #             print(count, "Visit Assign Attribute:", target.lineno, target.attr)
    #             count += 1
        # self.generic_visit(node)
    #     self.current_attr = None


    # def visit_Assign(self, node):
    #     if self.current_class:
    #         # print(self.current_class)
    #         # print(node.name)
    #         for target in node.targets:
    #             if isinstance(target, ast.Name):
    #                 self.classes[self.current_class]['variables'].append(target.id)
    #     elif hasattr(node, 'targets') and isinstance(node.targets[0], ast.Name):
    #         self.functions['variables'].append(node.targets[0].id)
    #     self.generic_visit(node)


def analyze_code(code):
    tree = ast.parse(code)
    analyzer = CodeAnalyzer()
    analyzer.visit(tree)
    return analyzer.classes, analyzer.functions, analyzer.ext_refs

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

# for python_file in python_files:
#     my_code = read_file_content(python_file)
#     print(python_file)
#     print(my_code)
#     classes, functions = analyze_code(my_code)
#     print("Classes:")
#     for class_name, class_data in classes.items():
#         print(f"Class: {class_name}")
#         print(f"Functions: {', '.join(class_data['functions'])}")
#         print(f"Variables: {', '.join(class_data['variables'])}")
#         print()
#     print("Functions:")
#     for function_name, function_data in functions.items():
#         print(f"Function: {function_name}")
#         print(f"Variables: {', '.join(function_data['variables'])}")
#         print()
        
my_code = read_file_content("/Users/zzyang/Documents/NNanim/TestingCode/vit.py")
# my_code = read_file_content("/Users/zzyang/Documents/NNanim/TestingCode/patch_embed.py")
# print(my_code)
classes, functions, ext_refs = analyze_code(my_code)
print("Classes:")
for class_name, class_data in classes.items():
    print(f"Class: {class_name}")
    print(f"Functions: {', '.join(class_data['functions'])}")
    print(f"Variables: {', '.join(class_data['variables'])}")
print()
print("Functions:")
for function_name, function_data in functions.items():
    print(f"Function: {function_name}")
    print(f"Variables: {', '.join(function_data['variables'])}")
print()
print("External Reference:")
for ext_ref_name, ext_ref_data in ext_refs.items():
    print(f"Packages: {ext_ref_name}")
    # file_path = get_module_file_path(ext_ref_name)
    # if file_path:
    #     print(f"Module: {ext_ref_name}, File Path: {file_path}")
    # else:
    #     print(f"Module: {ext_ref_name}, File Path not found")
    # print(f"Variables: {', '.join(ext_ref_data['variables'])}")