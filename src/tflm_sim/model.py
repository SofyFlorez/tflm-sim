"""This module contains the definition of the ModelError class and the Model class."""


import hjson # type: ignore[import-untyped]
from pathlib import Path
from prettytable import PrettyTable # type: ignore[import-not-found]

class ModelError(Exception):
    """Custom exception for model errors."""

class Model:
    """A class representing a model that can be read from a file."""

    UNSUPPORTED_FORMAT_MSG = "Format not supported"
    READ_ERROR_MSG = "Error reading JSON file"
    DECODE_ERROR_MSG = "Error decoding HJSON"


    def __init__(self, filename: str) -> None:
        """Initialize the Model instance by reading the model from the specified file.

        Read model from file 'filename' if the file is in one of the
        supported formats, raise ModelError exception otherwise.

        :param filename: The name of the file containing the model.
        :raises ModelError: If the file format is not supported or there is an error reading the file.
        """
        # Check file extension
        if not filename.endswith(".json"):
            raise ModelError(self.UNSUPPORTED_FORMAT_MSG)

        # Read JSON file using hjson library
        try:
            with Path(filename).open() as file:
                self._model_data = hjson.load(file)
        except OSError as e:
            raise ModelError(self.READ_ERROR_MSG) from e
        except hjson.HjsonDecodeError as e:
            raise ModelError(self.DECODE_ERROR_MSG) from e

    def summary(self) -> str:
        """Return a string with a summary of the model in tabular form."""
        table = PrettyTable()
        table.field_names = ["Operator Index", "Input Tensors", "Output Tensors", "Operator Type"]

        # Generate the table rows
        for subgraph in self._model_data.get("subgraphs", []):
            operators = subgraph.get("operators", [])
            for operator_idx, operator in enumerate(operators):
                opcode_index = operator.get("opcode_index", None)
                if opcode_index is not None:
                    operator_name = self._model_data["operator_codes"][opcode_index].get("builtin_code", "Unknown")
                else:
                    operator_name = "Unknown"
                input_tensors = operator.get("inputs", "Unknown")
                output_tensors = operator.get("outputs", "Unknown")
                table.add_row([operator_idx, input_tensors, output_tensors, operator_name])

        return table.get_string()

