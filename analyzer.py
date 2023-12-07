import ast, astor
from ast import Assign, Attribute, BinOp, Call, For, Name, Return, Tuple, mod
from gettext import find
import inspect
from re import L
from typing import Any
from numpy import isin, var
from sympy import false
from torch import rand
import torch.nn as nn
import random
import string
import numpy as np


class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    END = '\033[0m'
    
    
id_list = []


def generate_random_variable_name(length=8):
    '''
    Generates a random variable name.

    Parameters:
    - length (int): The length of the variable name. Default is 8.

    Returns:
    - str: A random variable name.
    '''
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_and_digits) for _ in range(length))


def generate_id(check_array, length=8):
    random_id = generate_random_variable_name(length=8)
    while random_id in check_array:
        random_id = generate_random_variable_name(length=8)
    return random_id


class AnalyzerSetups:
    def __init__(self,
                 analyze_depth=10,
                 debug_mod=False):
        self.analyze_depth = analyze_depth
        self.debug_mod = debug_mod
    

class ModuleAnalyzer:
    '''
    A class for analyzing the structure of PyTorch modules and their nested children.
    '''
    def __init__(self, analyzer_setup=AnalyzerSetups()):
        '''
        Initializes the ModuleAnalyzer.
        '''
        self.parameters = []                              #
        self.var_layers = []                              #
        self.l_flag = None                                      #
        self.v_flag = None                                      #
        self.analyzed_modules = {}                        #
        self.moudle_map = []                              #
        self.nn_module_flag = None                              #
        self.nn_module_index = None                             #
        self.nn_module_pack = []                          #
        self.debug_mod = False                                  #
        self.analyzer_setup = analyzer_setup    #
        
    def start_analyze_module(self, module):
        '''
        Initiates the analysis of a PyTorch module and its nested children.

        Parameters:
        - module (nn.Module): The PyTorch module to be analyzed.
        '''
        self.analyze_module("self", module)

    def analyze_module(self, var_name, module):
        '''
        Recursively analyzes the structure of PyTorch modules and their nested children.

        Parameters:
        - var_name (str): The variable name of the current module.
        - module (nn.Module): The PyTorch module to be analyzed.
        '''
        if isinstance(module, nn.Module):
            # Important
            # print(module.parameters)
            module_name = self.get_module_name(module)
            dot_var_name = self.get_var_name(var_name)
            self.var_layers.append((var_name, module_name))
            var_mod_list = {name:layer for name, layer in module.named_children()}
            # Print the name:class pair of the module
            # Red means first time analyzed + not torch built-in
            # Blue means the otherwise
            self.print_analyze_status(dot_var_name, module_name, module)
            if self.is_torch_module(module):
                self.update_module_flag(module)
                self.update_var_flag(var_name)
                lf_array = self.l_flag.split('.')
                vf_array = self.v_flag.split('.')
                lf_len = len(lf_array)
                vf_len = len(vf_array)
                # [We should make this another separated function]
                if self.nn_module_flag and self.nn_module_flag in dot_var_name:
                    op_in = self.moudle_map[self.nn_module_index][0]
                    op_out = self.moudle_map[self.nn_module_index][1]
                    self.nn_module_pack.append((op_in, op_out, f"{self.nn_module_flag}.{module_name}"))
                elif self.nn_module_flag and not self.nn_module_flag in dot_var_name:
                    # print("self.nn_module_pack", self.nn_module_flag, self.nn_module_pack)
                    self.moudle_map = self.update_module_map(self.nn_module_flag, self.nn_module_pack)
                    self.nn_module_flag = None
                    self.nn_module_index = None
                map_modules = [item[-1] for item in self.moudle_map]
                if var_name in map_modules:
                    self.nn_module_flag = var_name
                    self.nn_module_index = map_modules.index(var_name)
                    # op_in = self.moudle_map[self.nn_module_index][0]
                    # op_out = self.moudle_map[self.nn_module_index][1]
                    # self.nn_module_pack.append((op_in, op_out, f"{self.nn_module_flag}.{module_name}"))
                # [This function should ends here]
                for name, layer in module.named_children():
                    self.analyze_module(name, layer)
                if lf_len > 1:
                    self.l_flag = '.'.join(lf_array[:-1])
                else:
                    self.l_flag = None
                if vf_len > 1:
                    self.v_flag = '.'.join(vf_array[:-1])
                else:
                    self.v_flag = None
            # elif self.is_torch_module(module) and (module_name == 'Sequential' or module_name == 'ModuleList'):
            #     for name, layer in module.named_children():
            #         self.analyze_module(name, layer)
            else:
                # [This should be notice]
                if self.nn_module_flag and not self.nn_module_flag in dot_var_name:
                    # print("self.nn_module_pack", self.nn_module_flag, self.nn_module_pack)
                    self.moudle_map = self.update_module_map(self.nn_module_flag, self.nn_module_pack)
                    self.nn_module_flag = None
                    self.nn_module_index = None
                # [This should be notice]
                if module not in list(self.analyzed_modules.keys()):
                    module_code = inspect.getsource(type(module))
                    module_ast = ast.parse(module_code)
                    analyzer = ModuleAstAnalyzer(var_mod_list)
                    analyzer.visit(module_ast)
                    result = analyzer.module_map
                    self.moudle_map = self.update_module_map(var_name, result)
                else:
                    result = self.analyzed_modules[module_name]
                    self.moudle_map = self.update_module_map(var_name, result)
                print(f"{Color.GREEN}{self.moudle_map}{Color.END}")
                for name, layer in module.named_children():
                    self.analyze_module(name, layer)
                self.analyzed_modules[module_name] = analyzer.module_map
        return 0
            
    def update_module_map(self, var_name, replace_map):
        # Determine if our current module is mentioned in previous analysis
        # replace the original map with new analysis
        if len(self.moudle_map) == 0:
            return replace_map
        map_modules = [item[-1] for item in self.moudle_map]
        if var_name in map_modules and not len(replace_map) == 0:
            # Only replace the first appearance
            var_index = map_modules.index(var_name)
            indices = np.where(map_modules == var_name)[0]
            # for i in indices:
            #     self.moudle_map[i][-1] = f"{self.moudle_map[i][-1]}.{}"
            first_half = self.moudle_map[:var_index]
            second_half = self.moudle_map[var_index+1:]
            return first_half + replace_map + second_half
        else:
            return self.moudle_map
            
    def print_analyze_status(self, var_name, module_name, module):
        if module_name in self.analyzed_modules and self.is_torch_module(module):
            print(f"{Color.BLUE}({var_name}, {module_name}){Color.END}")
        else:
            print(f"{Color.RED}({var_name}, {module_name}){Color.END}")
                
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
        
    def get_var_name(self, v):
        if self.v_flag:
            return f"{self.v_flag}.{v}"
        else:
            return v
        
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
            
    def update_var_flag(self, v):
        if self.v_flag:
            self.v_flag = f"{self.v_flag}.{v}"
        else:
            self.v_flag = v
            
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
    def __init__(self, module_list):
        self.parent_stack = []              # Track the node visit history
        self.module_list:dict = module_list       # Here is all the nn.Module we wanna find in code
        self.module_map = []                # (Input, Output, Module)
        
        self.temp_var_ids = []
        self.forward_input = []
        self.current_var = ""
        
        self.forward_var_list = []
        self.forward_param_list = []
        self.analyzed_source_codes = set()
        self.out_flag = None
        self.out_dict = {}
        
    def generic_visit_with_parent_stack(self, node):
        self.parent_stack.append(node)
        self.generic_visit(node)
        self.parent_stack.pop()
        
    # Deal with functions and classes
        
    def visit_FunctionDef(self, node):
        # Only take look with how modules are called in nn.Module.forward
        if node.name == 'forward':
            for arg in node.args.args:
                # print("arguments in forward", arg.arg)
                if not arg.arg == 'self':
                    self.forward_var_list.append(arg.arg)
            self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        
    # Inside self.forward
    # analyze the nerual network structure
        
    def visit_Name(self, node: Name) -> Any:
        # if node.id in self.module_list:
        #     parent_type = [str(type(p)) for p in self.parent_stack]
        if node in self.parent_stack:
            return 0
        if node.id in self.forward_var_list:
            # parent_type = [str(type(p)) for p in self.parent_stack]
            self.analyze_net_name(self.parent_stack, node)
        self.generic_visit_with_parent_stack(node)
        
    def visit_Return(self, node: Return) -> Any:
        self.generic_visit(node)
        
    def visit_Call(self, node: Call) -> Any:
        self.generic_visit_with_parent_stack(node)
        
    def visit_For(self, node: For) -> Any:
        self.generic_visit_with_parent_stack(node)
        
    def visit_BinOp(self, node: BinOp) -> Any:
        self.generic_visit_with_parent_stack(node)
    
    def visit_Attribute(self, node: Attribute) -> Any:
        # If we found a module
        # a module must be an attribute !
        # well, we consider about other cases later
        # if node.attr in self.var_list:
            # parent_type = [str(type(p)) for p in self.parent_stack]
            # print(parent_type, node.attr)
            # prev_node = self.parent_stack[-1]
            # self.analyze_net_attr(self.parent_stack, node)
        # I think we have need to dig in
        self.generic_visit_with_parent_stack(node)
        
    def visit_Assign(self, node: Assign) -> Any:
        self.generic_visit_with_parent_stack(node)
        
    # Typer determination functions
    
    def find_full_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Attribute):
                return node.attr + '.' + self.find_full_name(node.value)
            elif isinstance(node.value, ast.Name) and not node.value.id == 'self':
                return node.attr + '.', node.value.id
            elif isinstance(node.value, ast.Name) and node.value.id == 'self':
                return node.attr
            elif isinstance(node.value, ast.Call):
                return node.attr
            else:
                pass # This won't happen
        # elif isinstance(node, ast.Call):
        #     return -'.'
            
    def find_all_names(self, node_list):
        ret = []
        for node in node_list:
            if isinstance(node, ast.Name) or isinstance(node, ast.Attribute):
                ret.append(self.find_full_name(node))
            elif isinstance(node, ast.Constant):
                ret.append(str(node.value))
            elif isinstance(node, ast.BinOp):
                ret.append(astor.to_source(node))
        return ret
        
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
        
    def analyze_net_attr(self, parents, this:Attribute):
        '''
        For what this function is dealing with:
        It must starts from an ast.Call: because every module is introduced by calling
        and it must end with an ast.Assign: because only a = b is something we interesting about
        '''
        # Reverse the parents array
        # Then analyze them one-by-one
        parents = parents[::-1]
        # The very first operartion must be the module we found
        current_var = None
        current_op = this.attr
        if len(parents) == 1:
            p = parents[0]
            if isinstance(p, ast.Call):
                op_in = [self.find_full_name(arg) for arg in p.args]
                # if op_in == [None]: op_in = [generate_id(self.temp_var_ids)]
                op_out = [generate_id(self.temp_var_ids)]
                self.temp_var_ids.append(op_out[0])
                tri_node = (op_in, op_out, current_op)
                self.module_map.append(tri_node)
                print(tri_node)
            else:
                parent_type = [str(type(p)) for p in parents]
                print(parent_type, this.attr)
        elif len(parents) == 2:
            p_call = parents[0]
            p_assign = parents[1]
            if isinstance(p_call, ast.Call) and p_call.func == this and isinstance(p_assign, ast.Assign) and p_assign.value == p_call:
                op_input = [self.find_full_name(arg) for arg in p_call.args]
                op_output = [self.find_full_name(target) for target in p_assign.targets]
                tri_node = (op_input, op_output, current_op)
                self.module_map.append(tri_node)
                print(tri_node)
                return 0
            else:
                parent_type = [str(type(p)) for p in parents]
                print(parent_type, this.attr)
        else:
            parent_type = [str(type(p)) for p in parents]
            print(parent_type, this.attr)
        # for i, p in enumerate(parents):
        # Return the final round of output variable
        return 0
    
    def visit_Tuple(self, node: Tuple) -> Any:
        return self.generic_visit_with_parent_stack(node)
    
    def analyze_net_name(self, parents, this:Name):
        if len(parents) == 0:
            return 0
        # print(f"{Color.PURPLE}{parents[0]}{self.analyzed_source_codes}{Color.END}")
        # if astor.to_source(parents[-1]) in self.analyzed_source_codes:
        #     return 0
        if parents[0] in self.analyzed_source_codes:
            return 0
        # At here, we made the parents upside down
        # so the above [0] is the future [-1]
        parents = parents[::-1]
        current_var = this.id
        # out_flag = None
        this_flag = this
        # current_modules = self.module_list
        if len(parents) == 1:
            p = parents[0]
            if isinstance(p, ast.Assign) and p.targets[0] == this:
                return 0
            parent_type = [str(type(p)) for p in parents]
            print(parent_type, this.id, astor.to_source(parents[0]))
        elif len(parents) == 2:
            p_0 = parents[0]
            p_1 = parents[1]
            if isinstance(p_0, ast.Call) and isinstance(p_1, ast.Assign):
                if self.find_full_name(p_0.func):
                    if self.find_full_name(p_0.func) in list(self.module_list.keys()):
                        op = self.module_list[self.find_full_name(p_0.func)]
                        op_in = [self.find_full_name(a) for a in p_0.args]
                        for oi in op_in:
                            if oi in list(self.out_dict.keys()):
                                op_in.remove(oi)
                                op_in.append(f'{oi}.{self.out_dict[oi]}')
                        op_id = generate_id(id_list)
                        id_list.append(op_id)
                        op_out = [f'{self.find_full_name(a)}.{op_id}' for a in p_1.targets]
                        for oo in op_out:
                            if oo.split('.')[0] in list(self.out_dict.keys()):
                                self.out_dict[oo.split('.')[0]] = oo.split('.')[1]
                        tri_node = (op_in, op_out, op)
                        self.module_map.append(tri_node)
                pass
            elif isinstance(p_0, ast.Attribute) and isinstance(p_1, ast.Assign) and isinstance(p_1.targets[0], ast.Tuple) and p_0.value == this:
                op = p_0.attr
                op_out = []
                target = p_1.targets[0]
                self.out_flag = this.id
                if isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            op_id = generate_id(id_list)
                            id_list.append(op_id)
                            op_out.append(f'{elt.id}.{op_id}')
                            # if not op == 'shape':
                            self.forward_param_list.append(elt.id)
                    op_id = generate_id(id_list)
                    id_list.append(op_id)
                    op_in = [f'{this.id}.{op_id}']
                    self.out_flag = f'{this.id}.{op_id}'
                    self.out_dict[this.id] = op_id
                    tri_node = (op_in, op_out, op)
                    self.module_map.append(tri_node)
                # for target in p_1.targets:
                #     print(target)
                #     if isinstance(target, ast.Name):
                #         op_id = generate_id(id_list)
                #         op_in = self.find_full_name(p_0.value)
                #         op_out = f'{target.id}.{generate_id(op_id)}'
                #         id_list.append(op_id)
                #         if not target.id in self.forward_var_list: self.forward_var_list.append(target.id)
                #         tri_node = (op_in, op_out, op)
                #         self.module_map.append(tri_node)
                #         print(self.module_map)
            else:
                pass
            parent_type = [str(type(p)) for p in parents]
            print(parent_type, this.id, astor.to_source(parents[-1]))
        else:
            for i, p in enumerate(parents):
                if isinstance(p, ast.Call):
                    # print(self.find_full_name(p.func))
                    if self.find_full_name(p.func):
                        if self.find_full_name(p.func) in list(self.module_list.keys()):
                            op = self.module_list[self.find_full_name(p.func)]
                            if self.out_flag:
                                op_in = [f'{self.out_flag}']
                            else:
                                op_in = None
                            if i < len(parents)-1:
                                op_id = generate_id(id_list)
                                id_list.append(op_id)
                                op_out = op_id
                                self.out_flag = op_out
                            else:
                                self.out_flag = None
                            tri_node = (op_in, [op_out], op)
                            self.module_map.append(tri_node)
                    pass
                elif isinstance(p, ast.Attribute):
                    if p.value == this_flag:
                        op = p.attr
                        if self.out_flag:
                            op_in = [f'{self.out_flag}']
                        else:
                            op_in = None
                        if i < len(parents)-1:
                            op_id = generate_id(id_list)
                            id_list.append(op_id)
                            op_out = op_id
                            self.out_flag = op_out
                        else:
                            self.out_flag = None
                        for oo in [op_out]:
                            if oo.split('.')[0] in list(self.out_dict.keys()):
                                self.out_dict[oo.split('.')[0]] = oo.split('.')[1]
                        tri_node = (op_in, [op_out], op)
                        self.module_map.append(tri_node)
                this_flag = p
            parent_type = [str(type(p)) for p in parents]
            print(parent_type, this.id, astor.to_source(parents[-1]))
        self.analyzed_source_codes.add(parents[-1])
        print(f"{Color.GREEN}{self.forward_var_list}{Color.END}")
        print(f"{Color.GREEN}{self.forward_param_list}{Color.END}")
        return 0
