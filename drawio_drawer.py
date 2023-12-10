import ast
from imp import new_module
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

def is_mm_in_hash_dict(mm, hash_var_dict):
    return mm in list(hash_var_dict.keys())

edge = []
edge_labels = []
new_module_map = []
unfinished = {}
unfinished_2 = {}
for mm in la.moudle_map:
    if len(mm[0]) > 1 or len(mm[1]) > 1:
        for a in mm[0]:
            for b in mm[1]:
                new_module_map.append((a, b, mm[2]))
    else:
        new_module_map.append((mm[0][0], mm[1][0], mm[2]))

print(new_module_map)
print(la.hash_var_dict)

finished = set()
count_x = 0
count_y = 0
for mm in new_module_map:
    if count_x == 40:
        count_x = 0
        count_y += 1
    if is_mm_in_hash_dict(mm[0], la.hash_var_dict) and is_mm_in_hash_dict(mm[1], la.hash_var_dict):
        print(la.hash_var_dict[mm[0]])
        diagram.add_node(label=la.hash_var_dict[mm[0]],
                             id=mm[0],
                             width=20,
                             height=120,
                             x_pos=count_x*80,
                             y_pos=count_y*240)
        count_x += 1
        print(la.hash_var_dict[mm[1]])
        diagram.add_node(label=la.hash_var_dict[mm[1]],
                             id=mm[1],
                             width=20,
                             height=120,
                             x_pos=count_x*80,
                             y_pos=count_y*240)
        count_x += 1
        diagram.add_link(mm[0], mm[1], label=mm[2])
        finished.update(mm[0])
        finished.update(mm[1])
        pass
    elif is_mm_in_hash_dict(mm[0], la.hash_var_dict) and (not is_mm_in_hash_dict(mm[1], la.hash_var_dict)):
        print(la.hash_var_dict[mm[0]])
        diagram.add_node(label=la.hash_var_dict[mm[0]],
                             id=mm[0],
                             width=20,
                             height=120,
                             x_pos=count_x*80,
                             y_pos=count_y*240)
        count_x += 1
        # if mm[1] in list(unfinished.keys()):
        unfinished[mm[1]] = mm[2]
        unfinished_2[mm[1]] = mm[0]
        pass
    elif (not is_mm_in_hash_dict(mm[0], la.hash_var_dict)) and is_mm_in_hash_dict(mm[1], la.hash_var_dict):
        if not mm[1] in finished:
            print(la.hash_var_dict[mm[1]])
            diagram.add_node(label=la.hash_var_dict[mm[1]],
                             id=mm[1],
                             width=20,
                             height=120,
                             x_pos=count_x*80,
                             y_pos=count_y*240)
            count_x += 1
        if mm[0] in list(unfinished_2.keys()) and mm[0] in list(unfinished.keys()):
            # if is_mm_in_hash_dict(unfinished_2[mm[0]], la.hash_var_dict):
            l = unfinished[mm[0]] + ", " + mm[2]
            ll = ''.join(l.split(', ='))
            diagram.add_link(unfinished_2[mm[0]], mm[1], label=ll)
        pass
    elif (not is_mm_in_hash_dict(mm[0], la.hash_var_dict)) and (not is_mm_in_hash_dict(mm[1], la.hash_var_dict)):
        if mm[0] in list(unfinished.keys()):
            unfinished[mm[1]] = unfinished[mm[0]] + ", " + mm[2]
            unfinished_2[mm[1]] = unfinished_2[mm[0]]
        pass
    # for a in mm[0]:
    #     if a in list(la.hash_var_dict.keys()):
    #         # print(la.hash_var_dict[a])
    #         diagram.add_node(label=la.hash_var_dict[a], id=a)
    #     else:
    #         unfinished[a] = str(mm[2])
    # for b in mm[1]:
    #     if b in list(la.hash_var_dict.keys()):
    #         # print(la.hash_var_dict[b])
    #         diagram.add_node(label=la.hash_var_dict[b], id=b)
    #     else:
    #         if b in list(unfinished.keys()):
                

            
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
# diagram.layout(algo="kk")
diagram.dump_file(filename="Attention_dim=2*768.drawio", folder="./Output/")
