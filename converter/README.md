# Circuitgraph Converter
This repository contains the base graph definition for the Circuitgraph software as well as converters for file import and export. Furthermore, image renderer and basic functionality for manipulating graphical graphs are provided.


## Structure
The folder structure is made up as follows:

```
gtdh-hd
│   README.md                   # This File
│   test.py                     # Illustrates Usage of all Converters
└───converter                   # The Actual Converter (Parser+Writer) Implementations
│   │   jsonConverter.py        # Internal (NotworkX) Graph Representation
│   │   pascalVocConverter.py   # Bounding Box (+Text, +Rotation) Annotations
│   │   labelmeConverter.py     # Polygon (+Text, +Rotation) Annotations
│   │   ...
└───core                        # Core Functionality
│   │   boundingbox.py          # Class Translating Between Internal Positioning and Bounding Boxes
│   │   converter.py            # The Converter Base Class
│   │   engineeringGraph.py     # EngGraph Class, the Central Definition of Graphical Graphs
│   │   renderer.py             # Helper Class for Turning an EngGraph into a List of Geometrical Primitives
└───symbols                     # Symbol-Related Data (a.k.a. Classes)
│   │   classes_ports.py        # Central Classes List
│   │   diode.json
│   │   resistor.json
│   │   fuse.json
│   │   ...
```

All converters inherit from the same `Converter` base class and return instances of `EngineeringGraph`. These are NetworkX Graphs which allow for geometric scaling and shifting of their contained nodes.

## Information Model
The individual modules of this software exchange circuits as graphs. Therefore, they must be denoted in a uniform way as given below. All Attributes are mandatory unless indicated otherwise.

### Graph Attributes
 - **name**: _str_
 - **width**: _int_
 - **height**: _int_
 - **image**: _str_

### Node Attributes
 - **id**: _int_
 - **name**: _str_
 - **type**: _str_
 - **position**: _position_
 - **text**: _str_ (Optional)
 - **ports**: _list[(str, position)]_ (Optional)
 - **shape**: _list[point]_ (Optional)
 - **properties**: _list[(str, any, position)]_ (Optional)

The attributes **id** and **name** have to be unique inside the graph. The list of allowed node `type` values can be found in `symbols/classes_ports.json`. The `rotation` describes the symbol orientation _within_ the non-rotated rectangle.

### Edge Attributes
 - **id**: int
 - **type**: _str_
 - **source**: _int_
 - **target**: _int_
 - **sourceConnector**: _int_ (Optional)
 - **targetConnector**: _int_ (Optional)
 - **shape**: _list[point]_ (Optional)
 
The list of allowed edge types includes:

 - `electrical` -> Wires, PCB-Wiring
 - `mechanical` -> complex relays, coupled button etc
 - `optical` -> optocouplers in which LED and Photodiode/Transistor are depicted far away from each other
 - `bus` -> multiple wires in one line
 - `property` -> relation between electronic components and their text boxes
 - `block` -> boundary of a functional block
 - `explainatory` -> helper to link text to component

### Position (Auxiliary Definiton)
 - **x**: _int_
 - **y**: _int_
 - **width**: _int_  (Optional if type==junction)
 - **height**: _int_ (Optional if type==junction)
 - **rotation**: _int_ Optional (Optional, expressed in degree)

## Legacy Port Definition
The following definitons are legacy only and will likely be updated/migrated in the upcoming version:








**classes_ports.json contains a list of component ports(the components are listed in classes.json)**

where:

for the bilateral components(the elements through which magnitude of current is independent of polarity of voltage like `resistor`, `capacitor.unpolarized`, `inductor`and ... ):

- `A` is the first port of the component.
- `B` is the second port of the component.

for the unilateral components(the direction of current is changed, the characteristics or property of element changes. like diodes) :
- `An` is anode.
- `Ca` is cathode.
- `G` is gate.
- `B` is base of a `transistor.bjt`.
- `E` is emmiter of a `transistor.bjt`.
- `C` is collector of `transistor.bjt`.
- `G` is gate of a `transistor.FET`.
- `S` is source of a `transistor.FET`.
- `D` is drain of a `transistor.FET`.

for `voltage.dc` source and `voltage.battery`:
- `P` denotes positive port.
- `N` denotes negative port.

for logic gates with two inputs:
- `A` and `B` denote inputs.
- `OUTPUT` denotes the output.

for the `transformer`:
- `A_1` denotes the first port of primary winding
- `A_2` denotes the second port of primary winding
- `B_1` denotes the first port of secondary winding
- `B_2` denotes the second port of secondary winding

remarks:
- currently the simplest type of transformer exists in the list. different kinds of transformers(multiple windings,tapped,multiple coils)can be added to the list of components therefore the ports configuration for them would be different.
- different kinds of `switch`es can be added to the components therefore port configuration of different switches can be defined:
`switch.SPST`, `switch.SPDT`, `switch.DPST` , `switch.DPDT` , `switch.push`, `switch.limit`, `switch.pressure`, `switch.flow`, `switch.temperature`, `switch.selector`, `switch.3PDT`
- different types of `relay`s have different port configurations and can be added to the list in the future.
- different types of `terminal`s and connectors have different port configurations and can be added to the list in the future.
- the general class of `integrated_circuit`s can be eliminated in `classes.json` and therefore different types of ICs with port configurations can be added to the port list.
- currently transistors are categorized in two main types `transistor.bjt` and `transistor.fet`. they can be expanded to different types(`transistor.bjt.pnp`, `transistor.bjt.npn`,`transistor.fet.nmos`,`transistor.fet.pmos` and ...)therefore ports can be defined differently.
