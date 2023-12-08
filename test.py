import numpy as np
import analyzer
from TestingCode import vit
from TestingCode import modules

# x = np.array([])
# vit.VisionTransformer(hidden_dims=[100]).forward(x)

la = analyzer.ModuleAnalyzer()
la.start_analyze_module(vit.VisionTransformer(embedding_dim=2*768, hidden_dims=[100]))
# la = analyzer.ModuleAnalyzer()
# la.start_analyze_module(modules.Attention(dim=2*768))

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