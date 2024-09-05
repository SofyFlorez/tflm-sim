"""This module contains the definition of the ModelError class and the Model class."""

import hjson  # type: ignore[import-untyped]
from pathlib import Path
from prettytable import PrettyTable  # type: ignore[import-not-found]
import numpy as np
import tensorflow as tf 

class ModelError(Exception):
    """Custom exception for model errors."""

class Model:
    """A class representing a model that can be read from a file."""

    UNSUPPORTED_FORMAT_MSG = "Format not supported"
    READ_ERROR_MSG = "Error reading JSON file"
    DECODE_ERROR_MSG = "Error decoding file"

    def __init__(self, filename: str) -> None:
        if not (filename.endswith(".json") or filename.endswith(".tflite")):
            raise ModelError(self.UNSUPPORTED_FORMAT_MSG)

        self.filename = filename
        self.input = None
        self.outputs = []
        self.is_tflite = filename.endswith(".tflite")

        if self.is_tflite:
            try:
                self.interpreter = self.build_tflite_native_interpreter(filename)
                self._init_data()  # Initialize data for TFLite model
            except Exception as e:
                raise ModelError(f"Error loading TFLite model: {str(e)}")
        else:  # JSON file
            try:
                with Path(filename).open() as file:
                    self._model_data = hjson.load(file)
                self._init_data()  # Initialize data for JSON model
            except OSError as e:
                raise ModelError(self.READ_ERROR_MSG) from e
            except hjson.HjsonDecodeError as e:
                raise ModelError(self.DECODE_ERROR_MSG) from e

    def _get_buffer(self, buffer_index: int) -> dict:
        """Retrieve the buffer data for a given buffer index."""
        buffers = self._model_data.get("buffers", [])
        if buffer_index >= len(buffers):
            raise ModelError(f"Buffer index {buffer_index} is out of range.")
        return buffers[buffer_index]

    def summary(self) -> str:
        """Return a string with a summary of the model in tabular form."""
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
        """Provide a detailed summary of a specific layer in a specific subgraph."""
        if subgraph >= len(self._model_data.get("subgraphs", [])):
            return f"Error: Subgraph index {subgraph} is out of range."
    
        subgraph_data = self._model_data["subgraphs"][subgraph]
    
        if layer >= len(subgraph_data.get("operators", [])):
            return f"Error: Layer index {layer} is out of range."

        layer_data = subgraph_data["operators"][layer]
        layer_type = self._get_layer_type(layer_data)

        summary = [f"Layer type: {layer_type}\n"]
        input_shape = self._get_layer_input_shape(subgraph, layer)
        for i, input_tensor_index in enumerate(layer_data.get("inputs", [])):
            input_tensor = self._get_tensor(subgraph, input_tensor_index)
            summary.append(self._format_tensor_summary(f"Input tensor #{i}", input_tensor, loc))

        output_shape = self._get_layer_output_shape(subgraph, layer)
        for i, output_tensor_index in enumerate(layer_data.get("outputs", [])):
            output_tensor = self._get_tensor(subgraph, output_tensor_index)
            summary.append(self._format_tensor_summary(f"Output tensor #{i}", output_tensor, loc))

        if layer_type == "CONV_2D":
            input_tensor = self._get_tensor(subgraph, layer_data["inputs"][0])
            kernel_tensor = self._get_tensor(subgraph, layer_data["inputs"][1])
            output_tensor = self._get_tensor(subgraph, layer_data["outputs"][0])
            summary.append(self._format_rescaling_factor(input_tensor, kernel_tensor, output_tensor))

        return "\n\n".join(summary)

    def _format_tensor_summary(self, tensor_name: str, tensor: dict, loc: tuple[int, int, int, int]) -> str:
        """Format the summary of a tensor."""
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
        """Format a grid of values as a string."""
        rows = []
        rows.append("            " + " ".join(f"{i:4d}" for i in range(values.shape[1])))
        for i, row in enumerate(values):
            rows.append(f"{i:4d} " + " ".join(f"{val:4.2f}" for val in row))
        return "\n".join(rows)

    def _get_layer_type(self, layer_data: dict) -> str:
        """Get the type of the layer based on its data."""
        opcode_index = layer_data.get("opcode_index", None)
        if opcode_index is not None:
            return self._model_data["operator_codes"][opcode_index].get("builtin_code", "Unknown")
        return "Unknown"

    def _get_tensor(self, subgraph: int, tensor_index: int) -> dict:
        """Retrieve a tensor from a subgraph."""
        tensors = self._model_data.get("subgraphs", [])[subgraph].get("tensors", [])
        if tensor_index >= len(tensors):
            raise ModelError(f"Tensor index {tensor_index} is out of range.")
        return tensors[tensor_index]

    def _get_layer_input_shape(self, subgraph: int, layer: int) -> dict:
        """Get the input shape of a layer."""
        layer_data = self._model_data["subgraphs"][subgraph]["operators"][layer]
        input_tensor_index = layer_data["inputs"][0]  # Assuming single input tensor
        tensor = self._get_tensor(subgraph, input_tensor_index)
        return tensor.get("shape", {})

    def _get_layer_output_shape(self, subgraph: int, layer: int) -> dict:
        """Get the output shape of a layer."""
        layer_data = self._model_data["subgraphs"][subgraph]["operators"][layer]
        output_tensor_index = layer_data["outputs"][0]  # Assuming single output tensor
        tensor = self._get_tensor(subgraph, output_tensor_index)
        return tensor.get("shape", {})

    def _format_rescaling_factor(self, input_tensor: dict, kernel_tensor: dict, output_tensor: dict) -> str:
        """Calculate and format the rescaling factor for a convolution layer."""
        input_scale = input_tensor.get('quantization', {}).get('scale', [1.0])[0]
        kernel_scale = kernel_tensor.get('quantization', {}).get('scale', [1.0])[0]
        output_scale = output_tensor.get('quantization', {}).get('scale', [1.0])[0]

        if input_scale and kernel_scale and output_scale:
            rescaling_factor = (input_scale * kernel_scale) / output_scale
            return f"\n  RESCALING FACTOR (for CONV_2D): {rescaling_factor:.4f}"
        return "\n  RESCALING FACTOR: Not available"

    def set_input_pattern(self, input_pattern: np.ndarray) -> None:
        """Set the input pattern."""
        if self.is_tflite:
            input_details = self.interpreter.get_input_details()[0]
            input_index = input_details['index']
            self.interpreter.set_tensor(input_index, input_pattern)
        else:
            raise ModelError("Setting input pattern is only supported for TFLite models.")

    def run_inference(self) -> np.ndarray:
        """Run inference and return the output."""
        if self.is_tflite:
            self.interpreter.invoke()
            output_details = self.interpreter.get_output_details()
            return [self.interpreter.get_tensor(out['index']) for out in output_details]
        else:
            raise ModelError("Running inference is only supported for TFLite models.")

    def build_tflite_native_interpreter(self, tflite_model_path: str) -> tf.lite.Interpreter:
        """Build and return a TFLite interpreter."""
        return tf.lite.Interpreter(model_path=tflite_model_path)
    
    def _init_data(self) -> None:
        """Initialize model data based on the file type (TFLite or JSON)."""
        if self.is_tflite:
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input = self.input_details[0]['index']  # Default to the first input
        else:
            # Extract input and output details from JSON model
            self.input = self._model_data.get('input', None)
            self.outputs = self._model_data.get('outputs', [])

    def _generate_pattern(self, shape, pattern, dtype, zero):
        """Generate a Numpy tensor with shape 'shape' and data type 'dtype' containing a
        data pattern useful for testing.

        Supported values of 'pattern':

            0:  All 'zero' in all channels.
            1:  All 'zero' in all channels except element (2,2) in all channels is 100-'zero'.
            2:  All 'zero' in all channels except elements (2+n*5,2+m*5) in all channels
            are 100-'zero', for n in [0 .. (height-1)//5], m in [0 .. (width-2)//5]
            (in other words, non-zero points each 5 pixels in both directions).
            TODO other patterns to be added

        Note that the 'zero' parameter is subtracted from all the values in the tensor,
        in accordance with TFLite's quantization scheme.

        returns a Numpy tensor.
        """
        # Initialize the tensor with zeros
        data = np.full(shape, zero, dtype=dtype)

        if pattern == 1:
            # Set a specific pattern
            data[2, 2] = 100 - zero
        elif pattern == 2:
            # Set a grid pattern
            height, width = shape[1], shape[2]
            for n in range(0, (height - 1) // 5 + 1):
                for m in range(0, (width - 2) // 5 + 1):
                    data[2 + n * 5, 2 + m * 5] = 100 - zero
        # Add additional patterns as needed

        return data



    def run_tflite_native_interpreter(self, interpreter: tf.lite.Interpreter, input: np.ndarray) -> None:
        """Run native TFLite interpreter with a given input tensor.

        Raises ModelError if the shape of the input tensor is not the shape expected
        by the model.

        Does not return anything but after execution the state of the interpreter
        object is modified and intermediate results can be examined.
        """
        # Ensure tensors are allocated
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        input_shape = tuple(input_details[0]['shape'])
        
        # Check if the input tensor shape matches the expected shape
        if input.shape != input_shape:
            raise ModelError(f"Input shape {input.shape} does not match expected shape {input_shape}")
        
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input)
        
        # Run inference
        interpreter.invoke()







