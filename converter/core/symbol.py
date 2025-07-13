"""symbol.py Symbol Management"""

__author__ = "Johannes Bayer, Vojtěch Sýkora"
__copyright__ = "Copyright 2022-2024, DFKI & RPTU & Others"
__status__ = "Development"

# System Imports
import json
from os import listdir, makedirs
from os.path import join
from typing import Dict, List, NamedTuple, Optional

# Project Imports
from converter.core.geometry import Circle, Line, Point, Polygon


class Port(NamedTuple):
    name: str
    position: Point


class Property(NamedTuple):
    name: str
    value: Optional[any]
    position: Optional[Point]


class Symbol(NamedTuple):
    name: str
    geometry: list
    ports: List[Port] = []
    properties: List[Property] = []


def load_symbols() -> Dict[str, Symbol]:
    """Loads all JSON Files from the Symbols Folder"""

    file_names = [
        file_name
        for file_name in listdir(join("converter", "symbols"))
        if file_name.endswith(".json")
    ]
    symbols = {}

    for file_name in file_names:
        with open(join("converter", "symbols", file_name)) as symbol_file:
            file_content = json.load(symbol_file)

            if type(file_content) is dict:
                symbol_name = ".".join(file_name.split(".")[:-1])
                symbol_geo = []

                for item in file_content.get("geometry", []):
                    if item.get("type", "") == "Line":
                        symbol_geo.append(
                            Line(
                                Point(item["a"]["x"], item["a"]["y"]),
                                Point(item["b"]["x"], item["b"]["y"]),
                                None,
                                None,
                                None,
                            )
                        )
                    if item.get("type", "") == "Circle":
                        symbol_geo.append(
                            Circle(
                                item["center"]["x"],
                                item["center"]["y"],
                                item["radius"],
                                None,
                                None,
                                item.get("fill", False),
                                None,
                            )
                        )
                    if item.get("type", "") == "Polygon":
                        symbol_geo.append(
                            Polygon(
                                [[point["x"], point["y"]] for point in item["points"]],
                                None,
                                None,
                                item.get("fill", False),
                                None,
                            )
                        )

                symbol_ports = [
                    Port(
                        port["name"],
                        Point(
                            float(port["position"]["x"]), float(port["position"]["y"])
                        ),
                    )
                    for port in file_content.get("ports", [])
                ]

                symbols[symbol_name] = Symbol(symbol_name, symbol_geo, symbol_ports)

    return symbols


def store_symbols(symbols: dict) -> None:
    """Stores all Entries of the Provided Dict as Separate JSON Files in the Symbols Folder"""
    for name, geo in symbols.items():
        with open(join("converter", "symbols", f"{name}.json"), "w") as symbol_file:
            json.dump(
                {
                    "geometry": [
                        {
                            "type": "Line",
                            "a": {"x": line.a.x, "y": line.a.y},
                            "b": {"x": line.b.x, "y": line.b.y},
                        }
                        for line in geo
                    ]
                },
                symbol_file,
                indent=4,
            )


def draw_symbol_geometry(ax, symbol: Symbol) -> None:
    """
    Draws the geometry elements and ports of a symbol on the given matplotlib axis.

    This helper function contains the shared drawing logic for a symbol.
    """
    from matplotlib.patches import Circle as mplCircle
    from matplotlib.patches import Polygon as mplPolygon

    # Draw geometry elements
    for geom in symbol.geometry:
        type_name = type(geom).__name__
        if type_name == "Line":
            ax.plot(
                [geom.a.x, geom.b.x],
                [geom.a.y, geom.b.y],
                color="black",
                linewidth=1,
            )
        elif type_name == "Circle":
            fill = getattr(geom, "fill", False)
            circle_patch = mplCircle(
                (geom.x, geom.y),
                geom.radius,
                edgecolor="black",
                facecolor="black" if fill else "none",
                linewidth=1,
            )
            ax.add_patch(circle_patch)
        elif type_name == "Polygon":
            fill = getattr(geom, "fill", False)
            poly_patch = mplPolygon(
                geom.points,
                closed=True,
                edgecolor="black",
                facecolor="black" if fill else "none",
                linewidth=1,
            )
            ax.add_patch(poly_patch)
        else:
            print(f"Unknown geometry type: {type_name}")

    # Draw ports as red circles with blue text labels.
    for port in symbol.ports:
        ax.plot(port.position.x, port.position.y, marker="o", color="red")
        ax.text(
            port.position.x,
            port.position.y,
            port.name,
            color="blue",
            fontsize=8,
            verticalalignment="bottom",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")


def render_symbol(
    symbol: Symbol, filename: str, dpi: int = 300, padding: float = 0.1
) -> None:
    """
    Renders the provided symbol as a PNG image with extra white padding.

    This function creates a matplotlib figure, calls the shared drawing function,
    and then saves the result as a PNG file.

    Parameters:
      symbol: The Symbol instance to render.
      filename: The output filename for the PNG image.
      dpi: The dots per inch setting for the output image.
      padding: Extra padding around the symbol in inches.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    draw_symbol_geometry(ax, symbol)
    plt.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=padding)
    plt.close(fig)


def render_all_symbols_collage(
    output_filename: str = "all_symbols.png",
    dpi: int = 300,
    cell_size: float = 3.0,
    cell_padding: float = 0.2,
) -> None:
    """
    Renders all symbols in one large image arranged in a grid so that you can view them all at once.

    Each symbol is drawn in its own cell with a bit of padding and has its name displayed above it.
    The grid layout is automatically computed to be nearly square.

    Parameters:
      output_filename: The PNG file to save the collage.
      dpi: The dots per inch setting for the output image.
      cell_size: The size in inches for each cell in the grid.
      cell_padding: Additional padding between cells (in inches).
    """
    from math import ceil, sqrt

    import matplotlib.pyplot as plt

    symbols = load_symbols()
    n_symbols = len(symbols)
    if n_symbols == 0:
        print("No symbols found!")
        return

    # Determine grid size (nearly square)
    cols = ceil(sqrt(n_symbols))
    rows = ceil(n_symbols / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * cell_size, rows * cell_size))
    if rows * cols == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    for ax in axs:
        ax.axis("off")

    sorted_symbols = sorted(symbols.items())
    for idx, (name, symbol) in enumerate(sorted_symbols):
        ax = axs[idx]
        draw_symbol_geometry(ax, symbol)
        ax.set_title(name, fontsize=10, pad=cell_padding * 72)

    # Hide any extra axes if present
    for ax in axs[len(sorted_symbols) :]:
        ax.set_visible(False)

    plt.tight_layout(pad=cell_padding)
    plt.savefig(output_filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # Render each symbol to its own PNG file.
    output_dir = join("converter", "symbols_png")
    makedirs(output_dir, exist_ok=True)
    symbols = load_symbols()
    for name, symbol in symbols.items():
        output_file = join(output_dir, f"{name}.png")
        print(f"Rendering symbol '{name}' to {output_file} ...")
        render_symbol(symbol, output_file)

    # Render a collage of all symbols into one big image.
    collage_file = join("converter", "symbols", "all_symbols.png")
    print(f"Rendering all symbols collage to {collage_file} ...")
    render_all_symbols_collage(collage_file)
    print("Done.")

    # Check which symbols are missing in the object detection classes
    symbols = load_symbols()
    print(f"Loaded {len(list(symbols.keys()))} symbols:")

    obj_det_classes = [
        "__background__",
        "text",
        "junction",
        "crossover",
        "terminal",
        "gnd",
        "vss",
        "voltage.dc",
        "voltage.ac",
        "voltage.battery",
        "resistor",
        "resistor.adjustable",
        "resistor.photo",
        "capacitor.unpolarized",
        "capacitor.polarized",
        "capacitor.adjustable",
        "inductor",
        "inductor.ferrite",
        "inductor.coupled",
        "transformer",
        "diode",
        "diode.light_emitting",
        "diode.thyrector",
        "diode.zener",
        "diac",
        "triac",
        "thyristor",
        "varistor",
        "transistor.bjt",
        "transistor.fet",
        "transistor.photo",
        "operational_amplifier",
        "operational_amplifier.schmitt_trigger",
        "optocoupler",
        "integrated_circuit",
        "integrated_circuit.ne555",
        "integrated_circuit.voltage_regulator",
        "xor",
        "and",
        "or",
        "not",
        "nand",
        "nor",
        "probe",
        "probe.current",
        "probe.voltage",
        "switch",
        "relay",
        "socket",
        "fuse",
        "speaker",
        "motor",
        "lamp",
        "microphone",
        "antenna",
        "crystal",
        "mechanical",
        "magnetic",
        "optical",
        "block",
        "unknown",
    ]
    symbols_names = list(symbols.keys())
    diff1 = set(symbols_names) - set(obj_det_classes)
    diff2 = set(obj_det_classes) - set(symbols_names)

    print("Symbols only in symbols_names:", diff1)
    print("Symbols only in obj_det_classes:", diff2)
    """
    Symbols only in symbols_names: {'rdf_function_classes', 'rdf_classes', 'classes_ports', 'kicad_classes', 'explanatory'}
    Symbols only in obj_det_classes: {'transistor.photo', 'transformer', 'triac', 'text', 'optical', '__background__', 'operational_amplifier.schmitt_trigger', 'inductor.coupled', 'unknown', 'magnetic', 'socket', 'optocoupler', 'mechanical', 'block', 'probe', 'relay', 'microphone'}
    """
