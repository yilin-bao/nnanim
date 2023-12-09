import hashlib
import random
import time
import numpy as np
import analyzer
from TestingCode import vit
from TestingCode import modules

# x = np.array([])
# vit.VisionTransformer(hidden_dims=[100]).forward(x)

# la = analyzer.ModuleAnalyzer()
# la.start_analyze_module(vit.VisionTransformer(embedding_dim=2*768, hidden_dims=[100]))
# print(list(la.var_module_dict.keys()))
# print(la.var_module_dict['transformer'])
# la = analyzer.ModuleAnalyzer()
# la.start_analyze_module(modules.Attention(dim=2*768))

def generate_random_string(length=8):
    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return ''.join(random.choice(characters) for _ in range(length))

def hash_code(code):
    code = code + generate_random_string()
    sha256_hash = hashlib.sha256()
    sha256_hash.update(code.encode('utf-8'))
    hashed_code = sha256_hash.hexdigest()
    return hashed_code

print(hash_code('x'))
print(hash_code('x'))

# print("="*40)

# for m in la.moudle_map:
#     print(m)
# for p in la.parameters:
#     print(p)

# current_parameters = dict(vit.VisionTransformer(hidden_dims=[100]).named_parameters())
# for name, param in current_parameters.items():
#     print(f"Parameter Name: {name}, Shape: {param.shape}")
# random_variable_name = analyzer.generate_random_variable_name()
# print(random_variable_name)
