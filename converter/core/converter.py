"""converter.py Converter Base Class"""

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2022-2023, DFKI"
__status__ = "Development"

# Project Imports
from converter.core.engineeringGraph import EngGraph


class Converter:
    """Converter Base Class"""

    STOREMODE = "w"

    def load(self, fileName: str) -> EngGraph:
        """Reads an Engineering Graph Object from a File"""

        with open(fileName) as fileStream:
            data = fileStream.read()

        return self._parse(data)

    def store(self, graph: EngGraph, fileName: str, **kwargs) -> None:
        """Writes an Engineering Graph Object to a File"""

        with open(fileName, self.STOREMODE) as fileStream:
            fileStream.write(self._write(graph, **kwargs))

        print(f"File {fileName} successfully written.")

    def _parse(self, data: str) -> EngGraph:
        """Reads a File String and Return an Engineering Graph Object"""

        print("PARSE METHOD NOT IMPLEMENTED")

        return EngGraph()

    def _write(self, graph: EngGraph, **kwargs) -> str:
        """Constructs a File String from an Engineering Graph"""

        print("WRITE METHOD NOT IMPLEMENTED")

        return ""
