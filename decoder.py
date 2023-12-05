import ast
from ast import Assign, mod
import inspect
from typing import Any
from numpy import var
from sympy import false
import torch.nn as nn

class ModuleAnalyzer:
    def __init__(self):
        self.parameters = []
        self.var_layers = []
        self.l_flag = None
        
    def start_analyze_module(self, module):
        self.analyze_module("", module)

    def analyze_module(self, var_name, module):
        if isinstance(module, nn.Module):
            module_name = self.get_module_name(module)
            self.var_layers.append((var_name, module_name))
            print((var_name, module_name))
            if self.is_torch_module(module):
                module_code = inspect.getsource(type(module))
                module_ast = ast.parse(module_code)
                analyzer = ModuleAstAnalyzer()
                analyzer.visit(module_ast)
            for name, layer in module.named_children():
                self.analyze_module(name, layer)
                
    def get_module_name(self, layer):
        if self.l_flag:
            return f"{self.l_flag}.{layer.__class__.__name__}"
        else:
            return layer.__class__.__name__
        
    def update_module_flag(self, layer):
        if self.l_flag:
            self.l_flag = f"{self.l_flag}.{layer.__class__.__name__}"
        else:
            self.l_flag = layer.__class__.__name__
            
    def is_torch_module(self, module):
        # Determine if this is a torch module (True)
        # or self-defined module (False)
        return module.__class__.__module__.startswith('torch')
    
    

class ModuleAstAnalyzer(ast.NodeVisitor):
    def __init__(self):
        pass
        
    def visit_FunctionDef(self, node):
        if node.name == 'forward':
            self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        
    def visit_Assign(self, node: Assign) -> Any:
        print(node.value)
        if self.all_NameAttribute(node.targets):
            self.generic_visit(node)
            
    def all_NameAttribute(self, targets):
        return self.all_Name(targets) or self.all_Attribute(targets)
        
    def all_Name(self, targets):
        for target in targets:
            if not isinstance(target, ast.Name):
                return False
        return True
    
    def all_Attribute(self, targets):
        for target in targets:
            if not isinstance(target, ast.Attribute):
                return False
        return True