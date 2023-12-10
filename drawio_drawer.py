import ast
import numpy as np
import analyzer as analyzer
from TestingCode import vit
from TestingCode import modules

# la = analyzer.ModuleAnalyzer()
# la.start_analyze_module(modules.Attention(dim=2*768))
la = analyzer.ModuleAnalyzer()
la.start_analyze_module(vit.VisionTransformer(embedding_dim=2*768, hidden_dims=[100]))

# print(la.moudle_map)
# print(la.hash_var_dict)

from N2G import drawio_diagram

diagram = drawio_diagram()
diagram.add_diagram("Page-1")

edge = []
edge_labels = {}
for mm in la.moudle_map:
    for a in mm[0]:
        if a in list(la.hash_var_dict.keys()):
            print(la.hash_var_dict[a])
            diagram.add_node(label=la.hash_var_dict[a], id=a)
        else:
            diagram.add_node(id=a)
    for b in mm[1]:
        if b in list(la.hash_var_dict.keys()):
            print(la.hash_var_dict[b])
            diagram.add_node(label=la.hash_var_dict[b], id=b)
        else:
            diagram.add_node(id=b)
    for a in mm[0]:
        for b in mm[1]:
            if mm[2] in list(la.var_module_dict.keys()):
                arr = la.var_module_dict[mm[2]].split('.')
                if len(arr) == 0:
                    l = ""
                elif len(arr) == 1:
                    l = arr[0]
                elif len(arr) > 1:
                    l = arr[-1]
                diagram.add_link(a, b, label=l)
            else:
                # print(mm[2])
                if isinstance(mm[2], ast.Add): diagram.add_link(a, b, label='Add')
                else: diagram.add_link(a, b, label=str(mm[2]))
            
    # a = (m_map[0][0], m_map[1][0])
    # b = m_map[2]
    # a_1 = f"{m_map[0][0]}"
    # a_2 = f"{m_map[1][0]}"
    # diagram.add_node(id=a_1)
    # diagram.add_node(id=a_2)
    # diagram.add_link(a_1, a_2, label=str(b))

# diagram.add_node(id="R1")
# diagram.add_node(id="R2")
# diagram.add_link("R1", "R2", label="DF", src_label="Gi1/1", trgt_label="GE23")
diagram.layout(algo="kk")
diagram.dump_file(filename="Attention_dim=2*768.drawio", folder="./Output/")
