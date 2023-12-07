import numpy as np
import analyzer
from TestingCode import vit
from TestingCode import modules

la = analyzer.ModuleAnalyzer()
la.start_analyze_module(modules.Attention(dim=2*768))

G_edge_form = []
edge_labels = {}
for m_map in la.moudle_map:
    # print(m_map)
    a = (m_map[0][0], m_map[1][0])
    G_edge_form.append(a)
    # print(a)
    edge_labels[a] = str(m_map[2]).split('\(')[0]
    
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
G.add_edges_from(G_edge_form)

pos = nx.spring_layout(G)

nx.draw(G, pos, cmap=plt.get_cmap('jet'), node_color='lightblue', with_labels=True, node_size=700, font_size=10, font_color="black")

# Add edge labels
# edge_labels = {('A', 'B'): 'Edge Label 1', ('A', 'C'): 'Edge Label 2', ('D', 'B'): 'Edge Label 3',
#                ('E', 'C'): 'Edge Label 4', ('E', 'F'): 'Edge Label 5',
#                ('B', 'H'): 'Edge Label 6', ('B', 'G'): 'Edge Label 7', ('B', 'F'): 'Edge Label 8',
#                ('C', 'G'): 'Edge Label 9'}

nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

plt.show()
