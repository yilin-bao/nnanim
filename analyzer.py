import ast, astor
import enum
from ctypes import Array
import time
import hashlib
from ast import Assign, Attribute, BinOp, Call, For, Name, Return, Subscript, Tuple, mod
from gettext import find
import inspect
from re import L
from typing import Any
from matplotlib.pylab import pareto
from numpy import isin, var
from sqlalchemy import String
from sympy import false
from torch import rand
import torch.nn as nn
import torch
import random
import string
import numpy as np

#============================================================
#================Here listed all [constants]=================
#============================================================

class Color:
    # ANSI escape codes for standard text colors
    BLACK = '\033[90m'
    RED = '\033[91m'                                        # for variable (names & attributes)
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'                                       # for module/layer (nn.Module, function & np attributes)
    PURPLE = '\033[95m'
    CYAN = '\033[96m'                                       # for numbers, and also torch shape and numpy dimension
    WHITE = '\033[97m'
    
    # Additional standard text colors
    LIGHT_GRAY = '\033[37m'
    DARK_GRAY = '\033[30m'
    
    # Extended set of standard text colors
    ORANGE = '\033[38;5;208m'
    PINK = '\033[38;5;200m'
    TEAL = '\033[38;5;51m'
    LIME = '\033[38;5;154m'
    BROWN = '\033[38;5;130m'
    GRAY = '\033[38;5;242m'
    
    # ANSI escape codes for bold text in various colors
    BOLD_BLACK = '\033[90;1m'
    BOLD_RED = '\033[91;1m'
    BOLD_GREEN = '\033[92;1m'
    BOLD_YELLOW = '\033[93;1m'
    BOLD_BLUE = '\033[94;1m'
    BOLD_PURPLE = '\033[95;1m'
    BOLD_CYAN = '\033[96;1m'
    BOLD_WHITE = '\033[97;1m'
    
    # Additional bold text colors
    BOLD_LIGHT_GRAY = '\033[37;1m'
    BOLD_DARK_GRAY = '\033[30;1m'
    
    # Extended set of bold text colors
    BOLD_ORANGE = '\033[38;5;214m'
    BOLD_PINK = '\033[38;5;197m'
    BOLD_TEAL = '\033[38;5;39m'
    BOLD_LIME = '\033[38;5;155m'
    BOLD_BROWN = '\033[38;5;94m'
    BOLD_GRAY = '\033[38;5;245m'
    
    # ANSI escape code for underlined text
    UNDERLINE = '\033[4m'

    # ANSI escape code to reset text attributes to default
    END = '\033[0m'
    
def print_ast_of_node(node):
    code = astor.to_source(node)
    tree = ast.parse(code)
    ast_code = astor.to_source(tree)
    print(f'{Color.BOLD_LIGHT_GRAY}{tree}{Color.END}')

def generate_random_string(length=8):
    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return ''.join(random.choice(characters) for _ in range(length))

def hash_code(code=" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):
    code = code + generate_random_string()
    sha256_hash = hashlib.sha256()
    sha256_hash.update(code.encode('utf-8'))
    hashed_code = sha256_hash.hexdigest()
    return hashed_code

#============================================================
#======================[Hash] functions======================
#============================================================

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
    """
    Generates a unique random identifier by repeatedly generating a random ID until
    it is not present in the given check_array.

    Parameters:
    - check_array (list): A list of existing IDs to check for uniqueness.
    - length (int): The length of the random ID. Default is 8.

    Returns:
    - str: A unique random identifier.
    """
    random_id = generate_random_variable_name(length=8)
    while random_id in check_array:
        random_id = generate_random_variable_name(length=8)
    return random_id

#============================================================
#===============Basic setups for the analyzers===============
#============================================================

class AnalyzerSetups:
    def __init__(self,
                 analyze_depth=10,
                 debug_mod=False):
        self.analyze_depth = analyze_depth
        self.debug_mod = debug_mod

#============================================================
#=============Dig all the nn.Module in the code==============
#============================================================

class ModuleAnalyzer:
    '''
    A class for analyzing the structure of PyTorch modules and their nested children.
    '''
    def __init__(self, analyzer_settings=AnalyzerSetups()):
        '''
        Initializes the ModuleAnalyzer.
        '''
        self.all_parameters = {}
        # var_module_mapping: Mapping of variables to PyTorch modules
        # An array of 2-tuples, each containing a variable name (first element) and its corresponding PyTorch module (second element).
        self.var_module_mapping = []
        self.var_module_dict = {}
        self.var_weight_dict = {}
        self.var_bias_dict = {}
        self.layer_flag = None
        self.var_flag = None
        self.moudle_map = []
        self.nn_module_flag = None
        self.nn_module_index = None
        self.nn_module_pack = []
        self.debug_mod = False
        # analyzer_settings: Object to store settings for all analyzers
        # An instance of a class that manages and stores settings for various analyzers.
        self.analyzer_settings: AnalyzerSetups = analyzer_settings
    
    def get_module_name(self, layer):
        '''
        Gets the full name of the module including its nested parent modules.

        Parameters:
        - layer (nn.Module): The PyTorch module.

        Returns:
        - str: The full name of the module.
        '''
        if self.layer_flag:
            return f"{self.layer_flag}.{layer.__class__.__name__}"
        else:
            return layer.__class__.__name__
        
    def get_var_name(self, name):
        if self.var_flag:
            return f"{self.var_flag}.{name}"
        else:
            return name
        
    def update_module_flag(self, layer):
        '''
        Updates the module flag to keep track of the current module's name.

        Parameters:
        - layer (nn.Module): The PyTorch module.
        '''
        if self.layer_flag:
            self.layer_flag = f"{self.layer_flag}.{layer.__class__.__name__}"
        else:
            self.layer_flag = layer.__class__.__name__
            
    def update_var_flag(self, name):
        if not name == 'self':
            if self.var_flag:
                self.var_flag = f"{self.var_flag}.{name}"
            else:
                self.var_flag = name
            
    def remove_module_flag(self):
        if self.layer_flag:
            arr = self.layer_flag.split('.')
            if len(arr) > 1:
                self.layer_flag = '.'.join(arr[:-1])
            else:
                self.layer_flag = None
            
    def remove_var_flag(self):
        if self.var_flag:
            arr = self.var_flag.split('.')
            if len(arr) > 1:
                self.var_flag = '.'.join(arr[:-1])
            else:
                self.var_flag = None
                
    def var_name_from_whole(self, var_name):
        arr = var_name.split('.')
        if len(arr) > 1:
            return arr[-1]
        elif len(arr) == 1:
            return arr[0]
        else:
            return var_name
    
    def start_analyze_module(self, module:nn.Module):
        '''
        Initiates the analysis of a PyTorch module and its nested children.

        Parameters:
        - module (nn.Module): The PyTorch module to be analyzed.
        '''
        self.all_parameters = dict(module.named_parameters())
        self.analyze_module("self", module)

    def analyze_module(self, var_name, module):
        '''
        Recursively analyzes the structure of PyTorch modules and their nested children.

        Parameters:
        - var_name (str): The variable name of the current module.
        - module (nn.Module): The PyTorch module to be analyzed.
        '''
        if isinstance(module, nn.Module):
            module_name = self.get_module_name(module)
            var_whole_name = self.get_var_name(var_name)
            self.print_current_layer_information(var_whole_name, module_name)
            self.var_module_mapping.append((var_whole_name, module_name))
            self.var_module_dict[var_whole_name] = (module_name)
            if f'{var_whole_name}.weight' in self.all_parameters:
                self.var_weight_dict[var_whole_name] = self.all_parameters[f'{var_whole_name}.weight'].shape
            if f'{var_whole_name}.bias' in self.all_parameters:
                self.var_bias_dict[var_whole_name] = self.all_parameters[f'{var_whole_name}.bias'].shape
            # var_mod_list = {name:layer for name, layer in module.named_children()}
            # Print the name:class pair of the module
            # self.print_analyze_status(var_whole_name, module_name, module)
            # print(var_name, var_whole_name)
            self.analyze_module_by_cases(var_whole_name, module_name, module)
            return var_whole_name, module_name
    
    def analyze_module_by_cases(self, var_name, module_name, module):
        # If anymore following work is needed here
        if self.is_torch_module(module):
            pass
        else:
            pass
        self.update_module_flag(module)
        self.update_var_flag(self.var_name_from_whole(var_name))
        var_module_layer = {}
        for name, layer in module.named_children():
            var_whole_name, sub_module_name = self.analyze_module(name, layer)
            var_module_layer[var_whole_name] = sub_module_name
        self.remove_module_flag()
        self.remove_var_flag()
        # Either if current module is a pyTorch in-built module
        # yes: then analyze inside sub-modules is meaningless
        # no: then we need to take a look with the logic
        if self.is_torch_module(module):
            self.analyze_inbuild_module()
        else:
            # print("self.layer_flag", self.var_flag)
            self.analyze_defined_module(var_name, module_name, module, var_module_layer)
        return 0
    
    def analyze_inbuild_module(self):
        return 0
    
    def analyze_defined_module(self, var_name, module_name, module, var_module_layer):
        module_code = inspect.getsource(type(module))
        module_ast = ast.parse(module_code)
        # [var_module_layer] is the variable-module dictionary of current layer
        analyzer = ModuleAstAnalyzer(var_module_layer, var_name, module_name)
        analyzer.visit(module_ast)
        # self.print_module_map(analyzer.module_map)
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
            first_half = self.moudle_map[:var_index]
            second_half = self.moudle_map[var_index+1:]
            return first_half + replace_map + second_half
        else:
            return self.moudle_map
            
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
    
    def print_analyze_status(self, var_name, module_name, module):
        if module_name in self.var_module_dict and self.is_torch_module(module):
            print(f"{Color.BLUE}({var_name}, {module_name}){Color.END}")
        else:
            print(f"{Color.RED}({var_name}, {module_name}){Color.END}")
            
    def print_current_layer_information(self, var_whole_name, module_name, depth=2):
        print('-'*60)
        arr = module_name.split('.')
        if len(arr) > depth:
            print(f"We have find a layer {Color.RED}{var_whole_name}{Color.END}, which is an instance of {Color.BLUE}{'.'.join(arr[-depth:])}{Color.END}")
        else:
            print(f"We have find a layer {Color.RED}{var_whole_name}{Color.END}, which is an instance of {Color.BLUE}{module_name}{Color.END}")
        if f'{var_whole_name}.weight' in self.all_parameters:
            # f"Parameter Name: {name}, Shape: {param.shape}"
            print(f"The weight tensor for this layer is {Color.CYAN}{self.all_parameters[f'{var_whole_name}.weight'].shape}{Color.END}")
        if f'{var_whole_name}.bias' in self.all_parameters:
            # f"Parameter Name: {name}, Shape: {param.shape}"
            print(f"The bias vector for this layer is {Color.CYAN}{self.all_parameters[f'{var_whole_name}.bias'].shape}{Color.END}")
        # else:
        #     print("This layer is not a final deconstructed layer, so there is no weight and bias")
    
    def print_module_map(self, module_map, length=8):
        print("================================")
        print("Analyzer returns the module_map:")
        print("================================")
        for i, mm in enumerate(module_map):
            print(f'Here is step {i} of the layer:', [a[:length] for a in mm[0]], [a[:length] for a in mm[1]], mm[2])
    
#============================================================
#======Ast static analyzer finds what happen in forward======
#============================================================

class ModuleAstAnalyzer(ast.NodeVisitor):
    def __init__(self, var_module_dict, var_name, module_name):
        # Parent stack, write in all the parents node visited before
        self.parent_stack = []
        # In [ModuleAstAnalyzer], [var_module_dict] is just the current analyzed layer
        self.var_module_dict:dict = var_module_dict
        self.var_name = var_name
        self.module_name = module_name
        self.module_map = [] # ([Inputs], [Outputs], Function/Operation)
        
        # forward_tensor_list: tensor, matrix, vector usd in deep learning
        # forward_param_list: int, float, dimension variables, other variables
        self.forward_tensor_list = []
        self.forward_param_list = []
        # Store the current version (by uuid) of each tensor/variable
        self.out_dict = {}
        self.hash_var_dict = {}
        
    #----------------------------------------
    #---------Generic visit enhance----------
    #----------------------------------------
        
    def generic_visit_with_parent_stack(self, node):
        self.parent_stack.append(node)
        self.generic_visit(node)
        self.parent_stack.pop()
        
    #----------------------------------------
    #----Boolean determination functions-----
    #----------------------------------------
        
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
    
    #----------------------------------------
    #---------Generically find names---------
    #----------------------------------------
    
    def find_full_name_array(self, nodes):
        if not nodes == None:
            ret = []
            for node in nodes:
                if isinstance(node, ast.Tuple):
                    for elt in node.elts:
                        elt_name = self.find_full_name(elt)
                        if elt_name:
                            ret.append(elt_name)
                elif isinstance(node, String):
                    if 'self' in node:
                        ret.append(node)
                else:
                    node_name = self.find_full_name(node)
                    if node_name:
                        ret.append(node_name)
            return ret
        return None
    
    def find_full_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f'{self.find_full_name(node.value)}.{node.attr}'
        else:
            return None
        
    def remove_starting_self(self, node_name):
        if node_name:
            arr = node_name.split('.')
            if arr[0] == 'self':
                return '.'.join(arr[1:])
            else:
                return '.'.join(arr)
    
    #--------------------------------------------------
    #--Analyze special functions (such as forward())---
    #--------------------------------------------------
        
    def visit_FunctionDef(self, node):
        # Only take look with how modules are called in nn.Module.forward
        if node.name == 'forward':
            for arg in node.args.args:
                # print("arguments in forward", arg.arg)
                if not arg.arg == 'self':
                    # self.forward_tensor_list = []
                    # self.forward_param_list = []
                    # self.out_dict = {}
                    # self.hash_var_dict = {}
                    self.forward_tensor_list.append(arg.arg)
                    arg_hashed = hash_code(arg.arg)
                    self.out_dict[arg.arg] = arg_hashed
                    self.hash_var_dict[arg_hashed] = arg.arg
            self.generic_visit(node)
            
    #--------------------------------------------------
    #----Rewrite some visting for history tracking-----
    #--------------------------------------------------

    def visit_Call(self, node: Call) -> Any:
        self.generic_visit_with_parent_stack(node)
        
    def visit_For(self, node: For) -> Any:
        self.generic_visit_with_parent_stack(node)
        
    def visit_BinOp(self, node: BinOp) -> Any:
        self.generic_visit_with_parent_stack(node)
    
    def visit_Attribute(self, node: Attribute) -> Any:
        self.generic_visit_with_parent_stack(node)
        
    def visit_Assign(self, node: Assign) -> Any:
        self.generic_visit_with_parent_stack(node)
        
    def visit_Tuple(self, node: Tuple) -> Any:
        self.generic_visit_with_parent_stack(node)
    
    def visit_Subscript(self, node: Subscript) -> Any:
        self.generic_visit_with_parent_stack(node)
    
    #--------------------------------------------------
    #-------------Core code of this class--------------
    #--------------------------------------------------
    
    def visit_Name(self, node: Name) -> Any:
        if node.id in self.forward_tensor_list or node.id in self.forward_param_list:
            self.analyze_net_ast(self.parent_stack, node)
        self.generic_visit_with_parent_stack(node)
        
    def analyze_net_ast(self, parents, this:Name):
        if len(parents) == 0:
            return 0
        # self.forward_tensor_list = []
        # self.forward_param_list = []
        # self.out_dict = {}
        # self.hash_var_dict = {}
        parents = parents[::-1]
        # ===
        self.print_parents_and_code(this)
        if len(parents) == 1:
            self.special_case_length_one(parents[0], this)
        else:
            current = None
            intermediate = None
            for parent in parents:
                if isinstance(parent, ast.Assign):
                    intermediate = self.analyze_net_ast_assign(parent, this, current, intermediate)
                elif isinstance(parent, ast.Call):
                    intermediate = self.analyze_net_ast_call(parent, this, current, intermediate)
                elif isinstance(parent, ast.Attribute):
                    intermediate = self.analyze_net_ast_attribute(parent, this, current, intermediate)
                elif isinstance(parent, ast.Subscript):
                    intermediate = self.analyze_net_ast_subscript(parent, this, current, intermediate)
                elif isinstance(parent, ast.For):
                    intermediate = self.analyze_net_ast_for(parent, this, current, intermediate)
                elif isinstance(parent, ast.BinOp):
                    intermediate = self.analyze_net_ast_binop(parent, this, current, intermediate)
                elif isinstance(parent, ast.Tuple):
                    intermediate = self.analyze_net_ast_tuple(parent, this, current, intermediate)
                current = parent
        # ===
        return 0
    
    def special_case_length_one(self, parent, this:Name):
        if isinstance(parent, ast.Attribute):
            # [<ast.Attribute object at 0x12aa0f790>] nn.Module
            # [<ast.Attribute object at 0x12aa0d4b0>] self.revised
            # self.print_parents_and_code(this)
            pass
        elif isinstance(parent, ast.Assign) and parent.targets[0] == this:
            # [<ast.Assign object at 0x130b28760>] x = self.embedding_layer(x)
            # [<ast.Assign object at 0x130b28940>] x = self.transformer(x)
            # [<ast.Assign object at 0x130b29a50>] x = self.post_transformer_ln(x)
            # [<ast.Assign object at 0x130b29f30>] x = self.cls_layer(x)
            # What we find is the variable [x] on the left side of assign, so ignore
            # self.print_parents_and_code(this)
            pass
        elif isinstance(parent, ast.Assign) and not parent.targets[0] == this:
            # seems is a case not gonna happen
            # self.print_parents_and_code(this)
            pass
        elif isinstance(parent, ast.Call):
            # self.print_parents_and_code(this)
            op_name = self.from_node_to_operation(parent.func)
            args = self.find_full_name_array(parent.args)
            self.update_module_name(args, args, op_name)
            pass
        else:
            # self.print_parents_and_code(this)
            pass
    
    def analyze_net_ast_assign(self, node:ast.Assign, this:Name, current, intermediate):
        print("Start analyzation on a node type of ast.Assign")
        op_name = "="
        if current:
            if node.value == current or node.targets[0] == current: # B, N, C = x.shape
                op_out = self.find_full_name_array(node.targets)
                if intermediate:
                    ret_hash = self.update_module_name_only_out([intermediate], op_out, op_name)[0]
                    return ret_hash
                else:
                    pass
            else:
                pass
        else:
            pass
        return intermediate
    
    def analyze_net_ast_call(self, node:ast.Call, this:Name, current, intermediate):
        print("Start analyzation on a node type of ast.Call")
        if current:
            if current in node.args:
                op_name = self.from_node_to_operation(node.func)
                ret_hash = self.update_module_name_hash_in([intermediate], op_name)[0]
                return ret_hash
            else:
                return intermediate
        else:
            op_name = self.from_node_to_operation(node.func)
            args = self.find_full_name_array(node.args)
            ret_hash = self.update_module_name_only_in(args, op_name)[0]
            return ret_hash
        return intermediate
    
    def analyze_net_ast_attribute(self, node:ast.Attribute, this:Name, current, intermediate):
        print("Start analyzation on a node type of ast.Attribute")
        op_name = node.attr
        if current:
            if node.value == current:
                ret_hash = self.update_module_name_hash_in([intermediate], op_name)[0]
                return ret_hash
            else:
                pass
        else:
            args = self.find_full_name_array([node.value])
            ret_hash = self.update_module_name_only_in(args, op_name)[0]
            return ret_hash
        return intermediate
    
    def analyze_net_ast_subscript(self, node:ast.Subscript, this:Name, current, intermediate):
        print("Start analyzation on a node type of ast.Subscript")
        if current:
            pass
        else:
            op_name = node.slice
            op_name = astor.to_source(op_name)
            args = [this.id + f'[{op_name}]']
            ret_hash = self.update_module_name_only_in(args, op_name)[0]
            return ret_hash
        return intermediate
    
    def analyze_net_ast_for(self, node:ast.For, this:Name, current, intermediate):
        print("Start analyzation on a node type of ast.For")
        if current:
            pass
        else:
            pass
        return intermediate
    
    def analyze_net_ast_binop(self, node:ast.BinOp, this:Name, current, intermediate):
        print("Start analyzation on a node type of ast.BinOp")
        op_name = node.op
        if current:
            pass
        else:
            args_left = self.find_full_name_array([node.left])
            args_right = self.find_full_name_array([node.right])
            print(args_left)
            print(args_right)
            ret_hash = self.update_module_name_only_in(args_left+args_right, op_name)[0]
            return ret_hash
        return intermediate
    
    def analyze_net_ast_tuple(self, node:ast.Tuple, this:Name, current, intermediate):
        print("Start analyzation on a node type of ast.BinOp")
        op_name = "="
        if current:
            pass
        else:
            args = self.find_full_name_array(node.elts)
            ret_hash = self.update_module_name_only_in(args, op_name)[0]
            return ret_hash
        return intermediate
        
    def is_parameter_or_tensor(self, op_name):
        return 0
        
    def from_node_to_operation(self, node):
        var_op = self.find_full_name(node)
        var_op_noself = self.remove_starting_self(var_op)
        if self.var_name == 'self':
            return var_op_noself
        else:
            return f'{self.var_name}.{var_op_noself}'
        
    def update_module_name_only_in(self, op_in_array:Array, op_name):
        op_in_hash = []
        for op_in in op_in_array:
            if op_in in self.out_dict:
                old_hash = self.out_dict[op_in]
            else:
                old_hash = hash_code(op_in)
            op_in_hash.append(old_hash)
        middle_hash = hash_code()
        op_out_hash = [middle_hash]
        operation = (op_in_hash, op_out_hash, op_name)
        self.module_map.append(operation)
        self.print_operation(operation)
        return [middle_hash]
    
    def update_module_name_hash_in(self, op_in_hash:Array, op_name):
        middle_hash = hash_code()
        op_out_hash: list[str] = [middle_hash]
        operation = (op_in_hash, op_out_hash, op_name)
        self.module_map.append(operation)
        self.print_operation(operation)
        return [middle_hash]
    
    def update_module_name_only_out(self, middle_hash, op_out_array:Array, op_name):
        op_out_hash = []
        for op_out in op_out_array:
            new_hash = hash_code(op_out)
            self.out_dict[op_out] = new_hash
            self.hash_var_dict[new_hash] = op_out
            op_out_hash.append(new_hash)
        operation = (middle_hash, op_out_hash, op_name)
        self.module_map.append(operation)
        self.print_operation(operation)
        return [op_out_hash]

    def update_module_name(self, op_in_array:Array, op_out_array:Array, op_name):
        if len(op_in_array) == 0:
            return None
        if len(op_out_array) == 0:
            return None
        op_in_hash = []
        for op_in in op_in_array:
            old_hash = self.out_dict[op_in]
            op_in_hash.append(old_hash)
        op_out_hash = []
        for op_out in op_out_array:
            new_hash = hash_code(op_out)
            self.out_dict[op_out] = new_hash
            self.hash_var_dict[new_hash] = op_out
            op_out_hash.append(new_hash)
        operation = (op_in_hash, op_out_hash, op_name)
        self.module_map.append(operation)
        self.print_operation(operation)
        return [op_out_hash]
    
    def update_module_name_hash(self, op_in_hash:Array, op_out_array:Array, op_name):
        if len(op_out_array) == 0:
            return None
        op_out_hash = []
        for op_out in op_out_array:
            new_hash = hash_code(op_out)
            self.out_dict[op_out] = new_hash
            self.hash_var_dict[new_hash] = op_out
            op_out_hash.append(new_hash)
        operation = (op_in_hash, op_out_hash, op_name)
        self.module_map.append(operation)
        self.print_operation(operation)
        return [op_out_hash]
    
    def print_parents_and_code(self, this:Name):
        if len(self.parent_stack) >= 1:
            print(f'{Color.BOLD_BLUE}{self.parent_stack}{Color.END} {this.id} {Color.LIME}{astor.to_source(self.parent_stack[0])}{Color.END}', end='')

    def print_operation(self, mm, length=8):
        if mm:
            print(f'Here is a layer of module map:', [a[:length] for a in mm[0]], [a[:length] for a in mm[1]], mm[2])
