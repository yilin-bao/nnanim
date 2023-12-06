import ast
from ast import Assign, Attribute, Call, For, Name, Return, mod
from gettext import find
import inspect
from typing import Any
from numpy import isin, var
from sympy import false
import torch.nn as nn


class ModuleAnalyzer:
    '''
    A class for analyzing the structure of PyTorch modules and their nested children.
    '''
    def __init__(self):
        '''
        Initializes the ModuleAnalyzer.
        '''
        self.parameters = []
        self.var_layers = []
        self.l_flag = None
        
    def start_analyze_module(self, module):
        '''
        Initiates the analysis of a PyTorch module and its nested children.

        Parameters:
        - module (nn.Module): The PyTorch module to be analyzed.
        '''
        self.analyze_module("", module)

    def analyze_module(self, var_name, module):
        '''
        Recursively analyzes the structure of PyTorch modules and their nested children.

        Parameters:
        - var_name (str): The variable name of the current module.
        - module (nn.Module): The PyTorch module to be analyzed.
        '''
        if isinstance(module, nn.Module):
            module_name = self.get_module_name(module)
            self.var_layers.append((var_name, module_name))
            var_mod_list = {name:layer for name, layer in module.named_children()}
            print((var_name, module_name))
            if self.is_torch_module(module):
                pass
            else:
                module_code = inspect.getsource(type(module))
                module_ast = ast.parse(module_code)
                analyzer = ModuleAstAnalyzer(var_mod_list)
                analyzer.visit(module_ast)
            for name, layer in module.named_children():
                self.analyze_module(name, layer)
                
    def get_module_name(self, layer):
        '''
        Gets the full name of the module including its nested parent modules.

        Parameters:
        - layer (nn.Module): The PyTorch module.

        Returns:
        - str: The full name of the module.
        '''
        if self.l_flag:
            return f"{self.l_flag}.{layer.__class__.__name__}"
        else:
            return layer.__class__.__name__
        
    def update_module_flag(self, layer):
        '''
        Updates the module flag to keep track of the current module's name.

        Parameters:
        - layer (nn.Module): The PyTorch module.
        '''
        if self.l_flag:
            self.l_flag = f"{self.l_flag}.{layer.__class__.__name__}"
        else:
            self.l_flag = layer.__class__.__name__
            
    def is_torch_module(self, module):
        '''
        Determines if the given module is a PyTorch module or a self-defined module.

        Parameters:
        - module (nn.Module): The PyTorch module.

        Returns:
        - bool: True if it is a PyTorch module, False otherwise.
        '''
        # Determine if this is a torch module (True)
        # or self-defined module (False)
        return module.__class__.__module__.startswith('torch')
    
    

class ModuleAstAnalyzer(ast.NodeVisitor):
    def __init__(self, var_list):
        self.parent_stack = []     # Track the node visit history
        self.var_list:dict = var_list    # Here is all the nn.Module we wanna find in code
        self.module_map = []       # (Input, Output, Module)
        
    def generic_visit_with_parent_stack(self, node):
        self.parent_stack.append(node)
        self.generic_visit(node)
        self.parent_stack.pop()
        
    # Deal with functions and classes
        
    def visit_FunctionDef(self, node):
        # Only take look with how modules are called in nn.Module.forward
        if node.name == 'forward':
            self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        
    # Inside self.forward
    # analyze the nerual network structure
        
    def visit_Name(self, node: Name) -> Any:
        if node.id in self.var_list:
            parent_type = [str(type(p)) for p in self.parent_stack]
        self.generic_visit_with_parent_stack(node)
        
    def visit_Return(self, node: Return) -> Any:
        self.generic_visit_with_parent_stack(node)
        
    def visit_Call(self, node: Call) -> Any:
        self.generic_visit_with_parent_stack(node)
        
    def visit_For(self, node: For) -> Any:
        self.generic_visit_with_parent_stack(node)
    
    def visit_Attribute(self, node: Attribute) -> Any:
        # If we found a module
        # a module must be an attribute !
        # well, we consider about other cases later
        if node.attr in self.var_list:
            parent_type = [str(type(p)) for p in self.parent_stack]
            print(parent_type, node.attr)
            # prev_node = self.parent_stack[-1]
            self.analyze_net_attr(self.parent_stack, node)
        # I think we have need to dig in
        # self.generic_visit_with_parent_stack(node)
        
    def visit_Assign(self, node: Assign) -> Any:
        self.generic_visit_with_parent_stack(node)
        
    # Typer determination functions
    
    def find_full_name(self, node: Attribute):
        if isinstance(node.value, ast.Attribute):
            return node.attr + self.find_full_name(node.value)
        elif isinstance(node.value, ast.Name):
            return node.attr + node.value.id
        else:
            pass # This won't happen
        
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
        
    def analyze_net_attr(self, parents, this):
        parents = parents[::-1]
        for p in parents:
            if isinstance(p, ast.Call) and p.func == this:
                self.module_map.append(())
                