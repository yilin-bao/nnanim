import numpy as np
import analyzer
from TestingCode import vit
from TestingCode import modules

la = analyzer.ModuleAnalyzer()
la.start_analyze_module(vit.VisionTransformer(hidden_dims=[100]))

from N2G import drawio_diagram

diagram = drawio_diagram()
diagram.add_diagram("Page-1")

G_edge_form = []
edge_labels = {}
for m_map in la.moudle_map:
    a = (m_map[0][0], m_map[1][0])
    b = m_map[2]
    a_1 = f"{m_map[0][0]}"
    a_2 = f"{m_map[1][0]}"
    diagram.add_node(id=a_1)
    diagram.add_node(id=a_2)
    diagram.add_link(a_1, a_2, label=str(b))

# diagram.add_node(id="R1")
# diagram.add_node(id="R2")
# diagram.add_link("R1", "R2", label="DF", src_label="Gi1/1", trgt_label="GE23")
diagram.layout(algo="kk")
diagram.dump_file(filename="Vit.drawio", folder="./Output/")