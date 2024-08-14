"""This module contains the definition of the ModelError class and the Model class."""

import hjson  # type: ignore[import-untyped]
from pathlib import Path
from prettytable import PrettyTable  # type: ignore[import-not-found]
import numpy as np

class ModelError(Exception):
    """Custom exception for model errors."""

class Model:
    """A class representing a model that can be read from a file."""

    UNSUPPORTED_FORMAT_MSG = "Format not supported"
    READ_ERROR_MSG = "Error reading JSON file"
    DECODE_ERROR_MSG = "Error decoding HJSON"

    def __init__(self, filename: str) -> None:
        """
        Initialize the Model instance by reading the model from the specified file.

        :param filename: The name of the file containing the model.
        :raises ModelError: If the file format is not supported or there is an error reading the file.
        """
        if not filename.endswith(".json"):
            raise ModelError(self.UNSUPPORTED_FORMAT_MSG)

        try:
            with Path(filename).open() as file:
                self._model_data = hjson.load(file)
        except OSError as e:
            raise ModelError(self.READ_ERROR_MSG) from e
        except hjson.HjsonDecodeError as e:
            raise ModelError(self.DECODE_ERROR_MSG) from e

        self.input = None
        self.outputs = []
        # Initialize internal tensors
        self._init_data

    def _get_buffer(self, buffer_index: int) -> dict:
        """Retrieve the buffer data for a given buffer index."""
        buffers = self._model_data.get("buffers", [])
        if buffer_index >= len(buffers):
            raise ModelError(f"Buffer index {buffer_index} is out of range.")
        return buffers[buffer_index]

    def summary(self) -> str:
        """
        Return a string with a summary of the model in tabular form.
        """
        table = PrettyTable()
        table.field_names = ["Operator Index", "Input Tensors", "Output Tensors", "Operator Type"]

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

    def layer_summary(self, subgraph: int, layer: int, loc: tuple[int, int, int, int]) -> str:
        """
        Provide a detailed summary of a specific layer in a specific subgraph.

        :param subgraph: Index of the subgraph.
        :param layer: Index of the layer.
        :param loc: Location (ochan, row, col, _).
        :return: Summary of the layer.
        """
        # Validate subgraph index
        if subgraph >= len(self._model_data.get("subgraphs")):
            return f"Error: Subgraph index {subgraph} is out of range."
    
        subgraph_data = self._model_data["subgraphs"][subgraph]
    
        if layer >= len(subgraph_data.get("operators", [])):
            return f"Error: Layer index {layer} is out of range."

        layer_data = subgraph_data["operators"][layer]

        # Determine layer type
        layer_type = self._get_layer_type(layer_data)

        summary = [f"Layer type: {layer_type}\n"]

        # Add input tensor summaries
        input_shape = self._get_layer_input_shape(subgraph, layer)
        for i, input_tensor_index in enumerate(layer_data.get("inputs", [])):
            input_tensor = self._get_tensor(subgraph, input_tensor_index)
            summary.append(self._format_tensor_summary(f"Input tensor #{i}", input_tensor, loc))

        # Add output tensor summaries
        output_shape = self._get_layer_output_shape(subgraph, layer)
        for i, output_tensor_index in enumerate(layer_data.get("outputs", [])):
            output_tensor = self._get_tensor(subgraph, output_tensor_index)
            summary.append(self._format_tensor_summary(f"Output tensor #{i}", output_tensor, loc))

        # Add specific details for CONV_2D layers
        if layer_type == "CONV_2D":
            input_tensor = self._get_tensor(subgraph, layer_data["inputs"][0])
            kernel_tensor = self._get_tensor(subgraph, layer_data["inputs"][1])
            output_tensor = self._get_tensor(subgraph, layer_data["outputs"][0])
            summary.append(self._format_rescaling_factor(input_tensor, kernel_tensor, output_tensor))

        return "\n\n".join(summary)

    def _format_tensor_summary(self, tensor_name: str, tensor: dict, loc: tuple[int, int, int, int]) -> str:
        """
        Format the summary of a tensor.

        :param tensor_name: Name of the tensor.
        :param tensor: Tensor data as a dictionary.
        :param loc: Location (ochan, row, col, _).
        :return: Formatted summary of the tensor.
        """
        summary = [
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",
            f" {tensor_name}\n",
            f"  SHAPE =       {tensor['shape']}",
            f"  ZERO POINT =  {tensor.get('quantization', {}).get('zero_point', 'N/A')}",
            f"  SCALE =       {tensor.get('quantization', {}).get('scale', 'N/A')}"
        ]

        buffer = self._get_buffer(tensor['buffer'])
        if 'data' in buffer and buffer['data']:
            try:
                data = np.array(buffer['data'], dtype=np.int32)
                expected_size = np.prod(tensor['shape'])
                if data.size != expected_size:
                    summary.append(f"\n  WARNING: Buffer data size ({data.size}) does not match expected tensor shape size ({expected_size})")
                    data = data.reshape(-1)
                else:
                    data = data.reshape(tensor['shape'])

                ochan, row, col, _ = loc
                window_size = 5
                half_window = window_size // 2

                if len(tensor['shape']) == 4:
                    if ochan >= tensor['shape'][3] or row >= tensor['shape'][1] or col >= tensor['shape'][2]:
                        summary.append("\n  RAW VALUES: Not available (invalid location for this tensor)")
                        summary.append("\n  ZERO-OFFSET VALUES: Not available (invalid location for this tensor)")
                    else:
                        row_start = max(0, row - half_window)
                        row_end = min(tensor['shape'][1], row + half_window + 1)
                        col_start = max(0, col - half_window)
                        col_end = min(tensor['shape'][2], col + half_window + 1)

                        raw_values = data[0, row_start:row_end, col_start:col_end, ochan]
                        zero_point = tensor.get('quantization', {}).get('zero_point', 0)
                        if isinstance(zero_point, (list, np.ndarray)):
                            zero_point = zero_point[ochan]
                        zero_offset_values = raw_values - zero_point

                        summary.append(f"\n  RAW VALUES around ({row},{col}) CHAN {ochan}:")
                        summary.append(self._format_value_grid(raw_values))

                        summary.append(f"\n  ZERO-OFFSET VALUES around ({row},{col}) CHAN {ochan}:")
                        summary.append(self._format_value_grid(zero_offset_values))
                else:
                    summary.append(f"\n  RAW VALUES for tensor with shape {tensor['shape']}:")
                    summary.append(self._format_value_grid(data.reshape(-1, data.shape[-1])[:5, :5]))
                    
                    zero_point = tensor.get('quantization', {}).get('zero_point', 0)
                    if isinstance(zero_point, (list, np.ndarray)):
                        zero_point = zero_point[0]
                    
                    zero_offset_values = data - zero_point
                    summary.append(f"\n  ZERO-OFFSET VALUES for tensor with shape {tensor['shape']}:")
                    summary.append(self._format_value_grid(zero_offset_values.reshape(-1, zero_offset_values.shape[-1])[:5, :5]))
            except Exception as e:
                summary.append(f"\n  ERROR: Failed to process buffer data: {str(e)}")
        else:
            summary.append("\n  RAW VALUES: Not available (buffer data is empty or missing)")
            summary.append("\n  ZERO-OFFSET VALUES: Not available (buffer data is empty or missing)")

        return "\n".join(summary)

    def _format_value_grid(self, values: np.ndarray) -> str:
        """
        Format a grid of values as a string.

        :param values: Grid of values as a numpy array.
        :return: Formatted grid of values as a string.
        """
        rows = []
        rows.append("            " + " ".join(f"{i:4d}" for i in range(values.shape[1])))
        rows.append("          " + "-" * (5 * values.shape[1]))
        for i, row in enumerate(values):
            rows.append(f"   {i:2d}    | " + " ".join(f"{val:4d}" for val in row))
        return "\n".join(rows)

    def _format_rescaling_factor(self, input_tensor: dict, kernel_tensor: dict, output_tensor: dict) -> str:
        """
        Format the rescaling factors of tensors.

        :param input_tensor: Input tensor data.
        :param kernel_tensor: Kernel tensor data.
        :param output_tensor: Output tensor data.
        :return: Formatted rescaling factor as a string.
        """
        summary = [" Composite rescaling factor:\n"]
        scale_in = input_tensor.get('quantization', {}).get('scale', 'N/A')
        scale_kernel = kernel_tensor.get('quantization', {}).get('scale', 'N/A')
        scale_out = output_tensor.get('quantization', {}).get('scale', 'N/A')

        summary.append(f"   SCALE IN:         {scale_in}")
        summary.append(f"   SCALE KERNEL:     {scale_kernel}")
        summary.append(f"   SCALE OUT:        {scale_out}")

        if isinstance(scale_in, (int, float)) and isinstance(scale_out, (int, float)):
            if isinstance(scale_kernel, (list, np.ndarray)):
                rescale_factors = [(scale_in * sk) / scale_out for sk in scale_kernel]
                summary.append(f"   RESCALE FACTORS:  {rescale_factors}")
            elif isinstance(scale_kernel, (int, float)):
                rescale_factor = (scale_in * scale_kernel) / scale_out
                summary.append(f"   RESCALE FACTOR:   {rescale_factor:.9f}")
            else:
                summary.append("   RESCALE FACTOR:   Unable to calculate (invalid kernel scale)")
        else:
            summary.append("   RESCALE FACTOR:   Unable to calculate (invalid input or output scales)")

        return "\n".join(summary)

    def _get_layer_type(self, layer_data: dict) -> str:
        """
        Determine the type of the layer based on its data.

        :param layer_data: Data for the layer.
        :return: The type of the layer as a string.
        """
        opcode_index = layer_data.get("opcode_index")
        if opcode_index is not None:
            return self._model_data["operator_codes"].get(opcode_index, {}).get("builtin_code", "Unknown")
    
        if "builtin_options" in layer_data:
            fused_activation = layer_data["builtin_options"].get("fused_activation_function")
            if fused_activation is not None:
                return "CONV_2D"  # Adjust this condition to match other layer types as needed

        return "Unknown"

    def _get_tensor(self, subgraph: int, tensor_index: int) -> dict:
        """
        Retrieve the tensor data from the subgraph.

        :param subgraph: Index of the subgraph.
        :param tensor_index: Index of the tensor.
        :return: Tensor data as a dictionary.
        """
        subgraph_data = self._model_data["subgraphs"][subgraph]
        tensor_data = subgraph_data["tensors"][tensor_index]
        return tensor_data

    def _get_layer_input_shape(self, graph: int, layer: int) -> tuple[int]:
        """
        Return the shape of the input feature of a given layer.

        :param graph: Index of the subgraph.
        :param layer: Index of the layer.
        :return: Shape of the input feature tensor.
        """
        subgraph = self._model_data["subgraphs"][graph]
        layer_data = subgraph["operators"][layer]
        input_tensor_index = layer_data["inputs"][0]
        input_tensor = self._get_tensor(graph, input_tensor_index)
        return tuple(input_tensor["shape"])

    def _get_layer_output_shape(self, graph: int, layer: int) -> tuple[int]:
        """
        Return the shape of the output feature of a given layer.

        :param graph: Index of the subgraph.
        :param layer: Index of the layer.
        :return: Shape of the output feature tensor.
        """
        subgraph = self._model_data["subgraphs"][graph]
        layer_data = subgraph["operators"][layer]
        output_tensor_index = layer_data["outputs"][0]
        output_tensor = self._get_tensor(graph, output_tensor_index)
        return tuple(output_tensor["shape"])

    def _zeros(self, shape: tuple) -> list:
        """
        Create a tensor filled with zeros with the given shape.

        :param shape: Shape of the tensor.
        :return: Tensor filled with zeros.
        """
        return [[[0 for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]

    def _init_data(self):
        """
        Initialize all the internal feature tensors to zero with the shapes and number of layers given by the model.
        """
        self.feature_tensors = []
        
        # Verify subgraphs and their tensors
        if "subgraphs" not in self._model_data:
            raise ModelError("No subgraphs found in model data.")
        
        subgraphs = self._model_data["subgraphs"]
        for subgraph in subgraphs:
            if "tensors" in subgraph:
                for tensor in subgraph["tensors"]:
                    shape = tensor.get("shape")
                    if shape:
                        self.feature_tensors.append(np.zeros(shape, dtype=np.float32))
                    else:
                        raise ModelError("Tensor shape not defined.")

        # Initialize input tensor if not already initialized
        if self.input is None:
            self.input = np.zeros((1, 28, 28, 1), dtype=np.float32)  


    def set_input_pattern(self, pattern: int): 
        """Initialize the input feature tensor with a simple data pattern."""
        if pattern == 0:
            self.input = np.zeros((1, 28, 28, 1))  # Create an all-zero tensor
            self.input[0, 1, 1, 0] = 100  # Set specific location to 100
        else:
            raise ValueError("Pattern value not supported. Only pattern 0 is supported.")

    def display_feature_window(self, subgraph_index: int, layer_index: int, row: int = 0, col: int = 0, channel: int = 0, height: int = 5, width: int = 5) -> str:
        """
        Return a string representation of a 'window' of a feature tensor.

        :param subgraph_index: Index of the subgraph.
        :param layer_index: Index of the layer.
        :param row: Row index of the window.
        :param col: Column index of the window.
        :param channel: Channel index of the window.
        :param height: Height of the window.
        :param width: Width of the window.
        :return: String representation of the feature window.
        """
        
        if layer_index < 0:
            tensor = self.input

         # FIXME: There may be an error here. The subgraph_index is compared with the length of self.outputs,
         # but it should be compared with the length of the subgraphs in self._model_data.   

        #elif subgraph_index < 0 or subgraph_index >= len(self.outputs):
            #return f"Error: Subgraph index {subgraph_index} is out of range."

        elif subgraph_index < 0 or subgraph_index >= len(self._model_data.get("subgraphs")):
            return f"Error: Subgraph index {subgraph_index} is out of range."

        # FIXME: Review this line as well for potential errors. It checks the layer_index but should align with the correct tensor structure.
        elif layer_index >= len(self.outputs[subgraph_index]):
            return f"Error: Layer index {layer_index} is out of range."
        else:
            tensor = self.outputs[subgraph_index][layer_index]

        if tensor is None or not isinstance(tensor, list):
            return "Error: Tensor is not valid."

        try:
            window = self._get_tensor_window(tensor, row, col, channel, height, width)
            return self._format_tensor_window(window)
        except Exception as e:
            return f"Error extracting feature window: {e}"

    def _get_tensor_window(self, tensor: list, row: int, col: int, channel: int, height: int, width: int) -> list:
        """
        Extract a window from a tensor.

        :param tensor: Tensor data as a list.
        :param row: Row index to start the window.
        :param col: Column index to start the window.
        :param channel: Channel index to extract.
        :param height: Height of the window.
        :param width: Width of the window.
        :return: Extracted window as a list.
        """
        window = []
        for i in range(max(0, row - height // 2), min(len(tensor), row + height // 2 + 1)):
            window_row = []
            for j in range(max(0, col - width // 2), min(len(tensor[0]), col + width // 2 + 1)):
                window_row.append(tensor[i][j][channel])
            window.append(window_row)
        return window

    def _format_tensor_window(self, window: list) -> str:
        """
        Format a tensor window as a string.

        :param window: Window data as a list.
        :return: Formatted window data as a string.
        """
        rows = [" ".join(f"{val:4d}" for val in row) for row in window]
        return "\n".join(rows)

    def execute(self):
        """
        Execute the model. Will update the internal output feature tensor variables in self.outputs.
        """
        # Initialize or reset all internal feature tensors
        self._init_data()
    
        """Execute the model."""
        if not self._model_data.get("subgraphs"):
            raise ModelError("No subgraphs available to execute.")
    
        # Execute each layer of the model
        try:
            for op_index, op in enumerate(self.operators):
                # Perform the operation based on its type
                if op.type == "MAX_POOL_2D":
                    self._perform_max_pool_2d(op_index)
                elif op.type == "RESHAPE":
                    self._perform_reshape(op_index)
                elif op.type == "FULLY_CONNECTED":
                    self._perform_fully_connected(op_index)
                elif op.type == "SOFTMAX":
                    self._perform_softmax(op_index)
                else:
                    print(f"Unknown operation type at index {op_index}.")
        
            # Post-processing to finalize outputs
            self._process_outputs()

        except Exception as e:
            print(f"Error during model execution: {e}")   

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Predict the output of the model given an input tensor.

        :param input_data: Input tensor data as a numpy array.
        :return: Output tensor data as a numpy array.
        """
        if not isinstance(input_data, np.ndarray):
            raise ValueError("Input data must be a numpy array.")

        # Ensure the input data has the correct shape
        expected_shape = self._get_layer_input_shape(0, 0)
        if input_data.shape != expected_shape:
            raise ValueError(f"Input data shape {input_data.shape} does not match expected shape {expected_shape}.")

        # Set the input tensor and execute the model
        self.input = input_data.tolist()
        self.execute()

        # For now, return the first subgraph's output as a placeholder
        return np.array(self.outputs[0][0])


