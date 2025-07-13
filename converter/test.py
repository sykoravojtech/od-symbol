"""test.py: Tests for the Converter Classes"""

# System Import
import os
from os.path import join
import subprocess

# Third-party Imports
import cv2

# Project Imports
from converter.converter.kicadConverter import KiCadConverter
from converter.converter.kicad6Converter import KiCad6Converter
from converter.converter.pascalVocConverter import PascalVocConverter
from converter.converter.rdfConverter import RdfConverter
from converter.converter.csvConverter import CsvConverter
from converter.converter.jsonConverter import JSONConverter
from converter.converter.pngConverter import PngConverter
from converter.converter.svgConverter import SvgConverter
from converter.converter.labelmeConverter import LabelMeConverter
from converter.converter.cocoConverter import CocoConverter


# Prepare Folder
if not os.path.exists("test"):
    os.mkdir("test")


# Test 0: Load LabelMe and store as image
c152_lm = join("gtdb-hd", "drafter_20", "instances_refined", "C236_D2_P2.json")
c152_img = join("gtdb-hd", "drafter_20", "images", "C236_D2_P2.jpeg")
graph = LabelMeConverter().load(c152_lm)

LabelMeConverter().store(graph, join("test", "C236_labelme.json"))
CocoConverter().store(graph, join("test", "C236_coco.json"))
SvgConverter().store(graph, join("test", "C236.svg"))
PngConverter().store(graph, join("test", "C236.png"))
PngConverter().store(graph, join("test", "C236_simple.png"), background=cv2.imread(c152_img))
SvgConverter().store(graph, join("test", "C236_complex.svg"),
                     bubble_scale=0.5, mode="complex", portDescription=True, nodeDescription=True, portCenter=True)
PngConverter().store(graph, join("test", "C236_complex.png"), background=cv2.imread(c152_img),
                     bubble_scale=0.5, mode="complex", portDescription=True, nodeDescription=True, portCenter=True)
PngConverter().store(graph, join("test", "C236_bg.png"), background=cv2.imread(c152_img), mode="complex")
PngConverter().store(graph, join("test", "C236_debug.png"), background=cv2.imread(c152_img), mode="complex",
                     nodeDescription=True, portDescription=True)
PngConverter().store(graph, join("test", "C236_mask.png"), mode="complex", shapes=True)
SvgConverter().store(graph, join("test", "C236_mask.svg"), mode="complex", shapes=True)
PngConverter().store(graph, join("test", "C236_mask_bg.png"), mode="complex", shapes=True,
                     drawNodeId=False, drawNodeType=False, background=cv2.imread(c152_img))


# Test 1: Load Pascal VOC and store in Various Formats
c152_voc = join("gtdb-hd", "drafter_13", "annotations", "C152_D1_P1.xml")
c152_img = join("gtdb-hd", "drafter_13", "images", "C152_D1_P1.jpg")
graph = PascalVocConverter().load(c152_voc)

for source, target in [(0, 13), (0, 16), (1, 4), (1, 15), (1, 17), (2, 14), (2, 15), (3, 15), (3, 19),
                       (4, 20), (5, 25), (5, 26), (6, 24), (7, 20), (7, 21), (8, 20), (8, 18),
                       (9, 21), (9, 22), (10, 12), (10, 23), (11, 23), (12, 21), (12, 24), (12, 26),
                       (13, 14), (16, 17), (17, 18), (18, 22), (19, 6), (19, 20), (22, 23), (24, 25)]:
    graph.add_edge(source, target, type="electrical")


KiCadConverter().store(graph, join("test", "C152.sch"))
RdfConverter().store(graph, join("test", "C152.ttl"))
CsvConverter().store(graph, join("test", "C152.csv"))
PngConverter().store(graph, join("test", "C152.png"))
PngConverter().store(graph, join("test", "C152_bg.png"), background=cv2.imread(c152_img), mode="complex")
JSONConverter().store(graph, join("test", "C152.json"))

# Test 2: Turtle Rendering
c177_voc = join("gtdb-hd", "drafter_15", "annotations", "C177_D1_P1.xml")
c177_img = join("gtdb-hd", "drafter_15", "images", "C177_D1_P1.jpeg")
graph = PascalVocConverter().load(c177_voc)

for source, target in [(45, 64), (47, 61), (46, 88), (36, 88), (36, 47), (80, 84), (30, 92), (30, 85),
                       (36, 89), (36, 49), (36, 54), (54, 57), (89, 90), (64, 90), (49, 65), (65, 70),
                       (59, 70), (58, 59), (58, 67), (59, 60), (45, 69), (60, 72), (53, 92), (56, 59),
                       (42, 69), (42, 72), (28, 67), (28, 72), (48, 67), (48, 53), (34, 53), (34, 69),
                       (27, 56), (27, 74), (27, 39), (39, 78), (35, 78), (29, 35), (29, 79), (79, 80),
                       (39, 85), (78, 86), (41, 86), (37, 41), (37, 68), (43, 68), (37, 38), (32, 38),
                       (14, 32), (14, 35), (24, 32), (24, 50), (50, 51), (51, 52), (31, 43), (43, 62),
                       (31, 51), (52, 77), (77, 82), (33, 52), (33, 71), (29, 71), (33, 75), (75, 79),
                       (40, 75), (40, 83), (40, 76), (80, 76), (30, 62)]:
    graph.add_edge(source, target, type="electrical")

PngConverter().store(graph, join("test", "C177.png"), background=cv2.imread(c177_img),
                     bubble_scale=0.5, mode="complex")
RdfConverter().store(graph, join("test", "C177.ttl"))

subprocess.run("cd insights && java -jar engine/target/circuit-inference-0.0.1.jar  --input=../test/C177.ttl --output=../test/C177_processed.ttl", shell=True)

graph = RdfConverter().load(join("test", "C177_processed.ttl"))
PngConverter().store(graph, join("test", "C177_processed.png"), background=cv2.imread(c177_img),
                     bubble_scale=0.5, mode="complex")



c182_voc = join("gtdb-hd", "drafter_16", "annotations", "C182_D1_P1.xml")
c182_img = join("gtdb-hd", "drafter_16", "images", "C182_D1_P1.jpg")
graph = PascalVocConverter().load(c182_voc)

for source, target in [(53, 77), (46, 77), (51, 53), (46, 70), (70, 101), (52, 101), (39, 96), (95, 105),
                       (18, 95), (61, 95), (51, 61), (48, 51), (48, 96), (62, 96), (52, 62), (18, 52), (29, 105),
                       (23, 29), (19, 29), (19, 26), (26, 44), (40, 44), (26, 66), (43, 66), (43, 50), (50, 97),
                       (23, 24), (24, 33), (33, 97), (33, 37), (33, 75), (37, 94), (90, 94), (50, 90), (69, 90),
                       (75, 89), (69, 89), (43, 49), (43, 76), (49, 98), (35, 98), (35, 42), (42, 84), (84, 88),
                       (28, 42), (28, 76), (22, 76), (22, 64), (24, 64), (76, 36), (25, 49), (25, 99), (30, 99),
                       (30, 35), (30, 54), (27, 54), (27, 59), (25, 59), (28, 85), (36, 82), (41, 82), (41, 72),
                       (41, 78), (78, 85), (85, 67), (38, 72), (38, 67), (38, 58), (38, 87), (56, 58), (86, 87),
                       (56, 91), (56, 86), (86, 102), (57, 102), (57, 80), (60, 73), (71, 73), (71, 100), (80, 100),
                       (68, 91), (55, 68), (47, 55), (47, 60), (55, 92), (32, 92), (32, 86), (32, 104), (80, 104),
                       (32, 93), (65, 93), (34, 47), (34, 65), (65, 79), (79, 100)]:
    graph.add_edge(source, target, type="electrical")

PngConverter().store(graph, join("test", "C182.png"), background=cv2.imread(c182_img),
                     bubble_scale=0.5, mode="complex")
RdfConverter().store(graph, join("test", "C182.ttl"))

subprocess.run("cd insights && java -jar engine/target/circuit-inference-0.0.1.jar  --input=../test/C182.ttl --output=../test/C182_processed.ttl", shell=True)

graph = RdfConverter().load(join("test", "C182_processed.ttl"))
PngConverter().store(graph, join("test", "C182_processed.png"), background=cv2.imread(c182_img),
                     bubble_scale=0.5, mode="complex")


c7_voc = join("gtdb-hd", "drafter_1", "annotations", "C7_D2_P4.xml")
c7_img = join("gtdb-hd", "drafter_1", "images", "C7_D2_P4.jpg")
graph = PascalVocConverter().load(c7_voc)

for source, target in [(1, 21), (1, 27), (25, 26), (26, 27), (20, 21), (14, 20),
                       (0, 11), (0, 12), (0, 14), (0, 15), (12, 13), (10, 13),
                       (3, 11), (3, 38), (15, 16), (5, 16), (5, 24), (24, 29),
                       (25, 29), (28, 29), (6, 29), (6, 16), (6, 17), (9, 18),
                       (26, 28), (4, 28), (4, 17), (17, 18), (8, 18), (8, 20),
                       (17, 19), (23, 28), (22, 23), (2, 23), (2, 19), (7, 19),
                       (7, 22), (7, 21)]:
    graph.add_edge(source, target, type="electrical")

import networkx as nx
nx.set_node_attributes(graph, {0: {"ports": [{'name': 'A', 'position': {'x': 1935, 'y': 2205}},
                                             {'name': 'B', 'position': {'x': 1960, 'y': 2500}},
                                             {'name': 'C', 'position': {'x': 2185, 'y': 2185}},
                                             {'name': 'D', 'position': {'x': 2195, 'y': 2460}}]},
                               1: {"ports": [{'name': 'A', 'position': {'x': 650, 'y': 1485}},
                                             {'name': 'B', 'position': {'x': 900, 'y': 1485}}]},
                               2: {"ports": [{'name': 'A', 'position': {'x': 600, 'y': 2150}},
                                             {'name': 'B', 'position': {'x': 810, 'y': 2140}}]},
                               3: {"ports": [{'name': 'A', 'position': {'x': 2400, 'y': 2045}},
                                             {'name': 'B', 'position': {'x': 2565, 'y': 2035}}]},
                               4: {"ports": [{'name': 'A', 'position': {'x': 600, 'y': 2550}},
                                             {'name': 'B', 'position': {'x': 770, 'y': 2525}}]},
                               5: {"ports": [{'name': 'A', 'position': {'x': 650, 'y': 3275}},
                                             {'name': 'B', 'position': {'x': 820, 'y': 3270}}]},
                               6: {"ports": [{'name': 'Collector', 'position': {'x': 1095, 'y': 2705}},
                                             {'name': 'Base', 'position': {'x': 980, 'y': 2900}},
                                             {'name': 'Emitter', 'position': {'x': 1085, 'y': 3070}}]},
                               7: {"ports": [{'name': 'Collector', 'position': {'x': 1075, 'y': 1700}},
                                             {'name': 'Base', 'position': {'x': 930, 'y': 1830}},
                                             {'name': 'Emitter', 'position': {'x': 1060, 'y': 1965}}]},
                               8: {"ports": [{'name': 'Positive', 'position': {'x': 1500, 'y': 1850}},
                                             {'name': 'Negative', 'position': {'x': 1495, 'y': 2000}}]},
                               28: {"ports": [{'name': 'A_1', 'position': {'x': 430, 'y': 2535}},
                                              {'name': 'A_2', 'position': {'x': 425, 'y': 2620}},
                                              {'name': 'B_1', 'position': {'x': 410, 'y': 2580}},
                                              {'name': 'B_2', 'position': {'x': 480, 'y': 2570}}]},
                               29: {"ports": [{'name': 'A_1', 'position': {'x': 430, 'y': 2895}},
                                              {'name': 'A_2', 'position': {'x': 420, 'y': 2995}},
                                              {'name': 'B_1', 'position': {'x': 420, 'y': 2935}},
                                              {'name': 'B_2', 'position': {'x': 490, 'y': 2935}}]}})

nx.set_edge_attributes(graph, {(0, 11): {"sourcePort": "C"}})
nx.set_edge_attributes(graph, {(0, 12): {"sourcePort": "D"}})
nx.set_edge_attributes(graph, {(0, 14): {"sourcePort": "A"}})
nx.set_edge_attributes(graph, {(0, 15): {"sourcePort": "B"}})
nx.set_edge_attributes(graph, {(1, 27): {"sourcePort": "A"}})
nx.set_edge_attributes(graph, {(1, 21): {"sourcePort": "B"}})
nx.set_edge_attributes(graph, {(2, 19): {"sourcePort": "B"}})
nx.set_edge_attributes(graph, {(2, 23): {"sourcePort": "A"}})
nx.set_edge_attributes(graph, {(3, 11): {"sourcePort": "A"}})
nx.set_edge_attributes(graph, {(3, 38): {"sourcePort": "B"}})
nx.set_edge_attributes(graph, {(4, 17): {"sourcePort": "B"}})
nx.set_edge_attributes(graph, {(4, 28): {"sourcePort": "A"}})
nx.set_edge_attributes(graph, {(5, 16): {"sourcePort": "B"}})
nx.set_edge_attributes(graph, {(5, 24): {"sourcePort": "A"}})
nx.set_edge_attributes(graph, {(6, 16): {"sourcePort": "Emitter"}})
nx.set_edge_attributes(graph, {(6, 17): {"sourcePort": "Collector"}})
nx.set_edge_attributes(graph, {(6, 29): {"sourcePort": "Base"}})
nx.set_edge_attributes(graph, {(7, 19): {"sourcePort": "Emitter"}})
nx.set_edge_attributes(graph, {(7, 21): {"sourcePort": "Collector"}})
nx.set_edge_attributes(graph, {(7, 22): {"sourcePort": "Base"}})
nx.set_edge_attributes(graph, {(8, 18): {"sourcePort": "Negative"}})
nx.set_edge_attributes(graph, {(8, 20): {"sourcePort": "Positive"}})
nx.set_edge_attributes(graph, {(4, 28): {"targetPort": "B_2"}})
nx.set_edge_attributes(graph, {(23, 28): {"targetPort": "A_1"}})
nx.set_edge_attributes(graph, {(26, 28): {"targetPort": "B_1"}})
nx.set_edge_attributes(graph, {(28, 29): {"sourcePort": "A_2"}})
nx.set_edge_attributes(graph, {(6, 29): {"targetPort": "B_2"}})
nx.set_edge_attributes(graph, {(24, 29): {"targetPort": "A_2"}})
nx.set_edge_attributes(graph, {(25, 29): {"targetPort": "B_1"}})
nx.set_edge_attributes(graph, {(28, 29): {"targetPort": "A_1"}})




KiCadConverter().store(graph, join("test", "C7.sch"))
RdfConverter().store(graph, join("test", "C7.ttl"))
CsvConverter().store(graph, join("test", "C7.csv"))
JSONConverter().store(graph, join("test", "C7.json"))
PngConverter().store(graph, join("test", "C7_simple.png"), background=cv2.imread(c7_img))
PngConverter().store(graph, join("test", "C7_complex.png"), background=cv2.imread(c7_img), mode="complex")
PngConverter().store(graph, join("test", "C7_debug.png"), background=cv2.imread(c7_img), mode="complex",
                     nodeDescription=True, portDescription=True)
print(JSONConverter().load(join("test", "C7.json")).nodes)

subprocess.run("cd insights && java -jar engine/target/circuit-inference-0.0.1.jar  --input=../test/C7.ttl --output=../test/C7_processed.ttl", shell=True)

graph = RdfConverter().load(join("test", "C7_processed.ttl"))
PngConverter().store(graph, join("test", "C7_processed.png"), background=cv2.imread(c7_img),
                     bubble_scale=0.5, mode="complex", portDescription=True)






c50_voc = join("gtdb-hd", "drafter_5", "annotations", "C50_D2_P1.xml")
c50_img = join("gtdb-hd", "drafter_5", "images", "C50_D2_P1.jpg")
graph = PascalVocConverter().load(c50_voc)

nx.set_node_attributes(graph, {13: {"ports": [{'name': "N", 'position': {'x': 545, 'y': 390}},
                                              {'name': "P", 'position': {'x': 550, 'y': 430}}]},
                               17: {"ports": [{'name': "A", 'position': {'x': 430, 'y': 400}},
                                              {'name': "B", 'position': {'x': 445, 'y': 440}}]},
                               20: {"ports": [{'name': "A", 'position': {'x': 200, 'y': 340}},
                                              {'name': "B", 'position': {'x': 255, 'y': 330}}]},
                               23: {"ports": [{'name': "A", 'position': {'x': 285, 'y': 365}},
                                              {'name': "B", 'position': {'x': 305, 'y': 430}}]},
                               24: {"ports": [{'name': "A", 'position': {'x': 260, 'y': 270}},
                                              {'name': "B", 'position': {'x': 250, 'y': 200}}]},
                               25: {"ports": [{'name': "A", 'position': {'x': 385, 'y': 165}},
                                              {'name': "B", 'position': {'x': 400, 'y': 240}}]},
                               26: {"ports": [{'name': "A", 'position': {'x': 520, 'y': 265}},
                                              {'name': "B", 'position': {'x': 570, 'y': 255}}]},
                               27: {"ports": [{'name': "B", 'position': {'x': 360, 'y': 330}},
                                              {'name': "C", 'position': {'x': 405, 'y': 290}},
                                              {'name': "E", 'position': {'x': 420, 'y': 360}}]}})

graph.add_edge(10, 28, type="electrical")
graph.add_edge(11, 26, targetPort='B', type="electrical")
graph.add_edge(12, 16, type="electrical")
graph.add_edge(13, 30, sourcePort='N', type="electrical")
graph.add_edge(13, 16, sourcePort='P', type="electrical")
graph.add_edge(14, 32, type="electrical")
graph.add_edge(15, 16, type="electrical")
graph.add_edge(15, 32, type="electrical")
graph.add_edge(15, 17, targetPort='B', type="electrical")
graph.add_edge(17, 31, sourcePort='A', type="electrical")
graph.add_edge(18, 19, type="electrical")
graph.add_edge(18, 23, targetPort='B', type="electrical")
graph.add_edge(18, 32, type="electrical")
graph.add_edge(20, 22, sourcePort='A', type="electrical")
graph.add_edge(20, 21, sourcePort='B', type="electrical")
graph.add_edge(21, 24, targetPort='A', type="electrical")
graph.add_edge(21, 23, targetPort='A', type="electrical")
graph.add_edge(21, 27, targetPort='B', type="electrical")
graph.add_edge(24, 33, sourcePort='B', type="electrical")
graph.add_edge(25, 28, sourcePort='A', type="electrical")
graph.add_edge(25, 29, sourcePort='B', type="electrical")
graph.add_edge(26, 29, sourcePort='A', type="electrical")
graph.add_edge(27, 29, sourcePort='C', type="electrical")
graph.add_edge(27, 31, sourcePort='E', type="electrical")
graph.add_edge(28, 33, type="electrical")
graph.add_edge(30, 31, type="electrical")


PngConverter().store(graph, join("test", "C50.png"), background=cv2.imread(c50_img), mode="complex",
                     drawNodeId=False, drawNodeType=False, portDescription=False, bubble_scale=0.3)
PngConverter().store(graph, join("test", "C50_debug.png"), background=cv2.imread(c50_img), mode="complex",
                     drawNodeId=False, drawNodeType=False, portDescription=True, bubble_scale=0.3)
RdfConverter().store(graph, join("test", "C50.ttl"))

subprocess.run("cd insights && java -jar engine/target/circuit-inference-0.0.1.jar  --input=../test/C50.ttl --output=../test/C50_processed.ttl", shell=True)

graph = RdfConverter().load(join("test", "C50_processed.ttl"))
PngConverter().store(graph, join("test", "C50_processed.png"), background=cv2.imread(c50_img),
                     drawNodeId=False, drawNodeType=False, bubble_scale=0.5, portDescription=False,
                     mode="complex")




# Test 3: KiCad Loading
graph = KiCadConverter().load(join("gtdb-cad", "circuits", "C24.sch"))
graph *= 1/5
JSONConverter().store(graph, join("test", "C24.json"))
PascalVocConverter().store(graph, join("test", "C24.xml"))
PngConverter().store(graph, join("test", "C24.png"),
                     bubble_scale=0.5, mode="complex", portDescription=True, nodeDescription=True, portCenter=True)


# Test 4: KiCad Loading + Streching
graph = KiCadConverter().load(join("gtdb-cad", "circuits", "C25.sch"))
graph *= 1/5, 1/4
JSONConverter().store(graph, join("test", "C25.json"))
PascalVocConverter().store(graph, join("test", "C25.xml"))
PngConverter().store(graph, join("test", "C25.png"),
                     bubble_scale=0.5, mode="complex", portDescription=True, nodeDescription=True, portCenter=True)




def process_rdf(graph):

    RdfConverter().store(graph, join("test", f"{c}.ttl"))
    subprocess.run(
        f"cd insights && java -jar engine/target/circuit-inference-0.0.1.jar  --input=../test/{c}.ttl --output=../test/{c}_processed.ttl",
        shell=True)
    graph = RdfConverter().load(join("test", f"{c}_processed.ttl"))

    return graph


for i in ["microcontroller"]:#range(1, 204):  # "microcontroller", "FlybackDiod", "test"
    c = f"{i}"
    print(c)
    graph = KiCadConverter().load(join("gtdb-cad", "circuits", c + ".sch"))
    import networkx as nx
    print(nx.get_node_attributes(graph, "type"))
    #PngConverter().store(graph, join("test", c + ".png"),
    #                     bubble_scale=0.5, mode="complex", portDescription=True, nodeDescription=True, portCenter=True)
    #KiCadConverter().store(graph, join("test", c + ".sch"))
    #JSONConverter().store(graph, join("test", f"{c}.json"))
    graph = process_rdf(graph)
    graph *= 0.3
    #JSONConverter().store(graph, join("test", f"{c}_processed.json"))
    PngConverter().store(graph, join("test", f"{c}_processed.png"),
                         bubble_scale=0.5, mode="complex", portDescription=True)








# l1=[1,2,3,4,5,6,7]
# print(l1[1:-1])
# print(l1[::2])
# for x, y in zip(l1[::2], l1[1::2]):
#     print(x,y)
# l2='"V?"'
# print(l2[1:-1])
#
# x, y, rotation= line.split(" ")[6:9]
# rotation=int(rotation[:-1])
# x=float(x)
# y=float(y)
# print(x+y,rotation)
import os

# line='    (in_bom yes) (on_board yes)'
# if line.startswith("    (in_bom"):
#     in_bom = line.split(" ")[5]
#     in_bom=in_bom[:-1]
#     in_bom=True if in_bom=="yes" else False
#     print(in_bom)


# with open('/home/jaikrishna/PycharmProjects/Circuitgraph/gtdb-cad/circuits/C3.kicad_sch') as fileStream:
#     data = fileStream.read()

# lines = data.splitlines()
# list_lines=list(enumerate(lines))
# for line_number,line in list_lines:
#
#     if line.startswith("(kicad_sch"):
#         head_start=line_number
#         #print("Head start: ",head_start)
#     if line.startswith("  (lib_symbols"):
#         head_end=line_number-1
#         symbol_definitions_start = line_number
#         # print("Head end: ",head_end)
#         # print("symbol def start: ",symbol_definitions_start)
#
#
#
# for line_number,line in list_lines:
#     if line.startswith("  (wire (pts"):
#         symbol_definitions_end=line_number-1
#         # print("symbol def end: ",symbol_definitions_end)
#         wires_start = line_number
#         # print("wires_start: ",wires_start)
#         break
#
#
# for line_number,line in list_lines:
#     if line.startswith("  (symbol (lib_id"):
#         wires_end=line_number-1
#         # rint("wires_end: ",wires_end)
#         symbols_start = line_number
#         # print("symbols_start: ",symbols_start)
#         break
#
# for line_number,line in list_lines:
#     if line.startswith("  (sheet_instances"):
#         symbols_end=line_number-1
#         # print("symbols_end: ",symbols_end)
#         sheet_start = line_number
#         # print("sheet_start: ",sheet_start)
#
#     if line.startswith("  (symbol_instances"):
#         sheet_end=line_number-1
#         # print("sheet_end: ",sheet_end)
#         symbol_instances_start = line_number
#         # print("symbol_instances_start: ",symbol_instances_start)
#
# head_list=list_lines[head_start:head_end]
# symbol_definitions_list=list_lines[symbol_definitions_start:symbol_definitions_end]
# wires_list=list_lines[wires_start:wires_end]
# symbols_list=list_lines[symbols_start:symbols_end]
# sheet_list=list_lines[sheet_start:sheet_end]
# symbol_instances_list=list_lines[symbol_instances_start:]


# print(symbol_instances_list)


#
#
#     if not found and lib_line.startswith(component_head):
#         found = True
#     if not found and lib_line.startswith("ALIAS") and (component in lib_line):
#         found = True
#     if found:
#         if lib_line.startswith("X"):
#             lib_line_parts = lib_line.split(" ")
#             name = lib_line_parts[2]
#             x = int(lib_line_parts[3])
#             y = -int(lib_line_parts[4])
#             ports[name] = (x, y)
#             bb.update(x, y)
#         if lib_line.startswith("P "):
#             raw = [int(value) for value in lib_line.split(" ")[1:-1]]
#             for x, y in zip(raw[::2], raw[1::2]):
#                 bb.update(x, y)
#         if lib_line.startswith("S "):
#             lib_line_parts = lib_line.split(" ")
#             bb.update(int(lib_line_parts[1]), int(lib_line_parts[2]))
#             bb.update(int(lib_line_parts[3]), int(lib_line_parts[4]))
#         if lib_line.startswith("ENDDEF"):
#             break
#
# return ports, bb
import networkx as nx
graph = KiCad6Converter().load(os.path.join("gtdb-cad", "circuits", "C2.kicad_sch"))
#graph *= float(1/5)


# [{"key": property_key, "value": property_value, "id": property_id, "x": property_x, "y": property_y,
#                   "rotation": property_rotation}]

#print(nx.get_node_attributes(graph, 'properties'))
#print([e for e in graph.edges])
# for _, _, attrs in graph.edges.data():
#     print(attrs)
# node_prop = nx.get_node_attributes(graph, 'properties')
# node_type = nx.get_node_attributes(graph, 'type')
# symbol_instances={}
# reference=''
# value=''
# footprint=''
# for node in graph.nodes:
#     for property in node_prop[node]:
#         if property['key']=='"Reference"':
#             reference=property['value']
#         if property['key']=='"Value"':
#             value=property['value']
#         if property['key']=='"Footprint"':
#             footprint=property['value']
#
#     if node_type[node]!='junction':
#         symbol_instances[node]={'reference': reference, 'unit': 1, 'value': value,'footprint': footprint}
# print(symbol_instances)




KiCad6Converter().store(graph, os.path.join("test", "C2.kicad_sch"),
                        bubble_scale=0.5, mode="complex", portDescription=True,
                        background=cv2.imread(join("gtdb-cad","png", f"{c}.png")))
