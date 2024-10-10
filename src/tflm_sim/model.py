"""This module contains the definition of the ModelError class and the Model class."""


from pathlib import Path
from prettytable import PrettyTable  # type: ignore[import-not-found]
import numpy as np
import tensorflow as tf
from tflite.Model import Model as TfliteModel
from tflite.BuiltinOperator import BuiltinOperator
from tflite.Conv2DOptions import Conv2DOptions
from tflite.Pool2DOptions import Pool2DOptions
from tflite.FullyConnectedOptions import FullyConnectedOptions
import flatbuffers  


class ModelError(Exception):
    """Custom exception for model errors."""


def _extract_operator_names():
    """Helper function to create a mapping from opcode indices to operator names."""
    opcode_to_opname = {}
    for k in BuiltinOperator.__dict__.keys():
        if not k.startswith("_"):
            opcode_to_opname[BuiltinOperator.__dict__[k]] = k
    return opcode_to_opname


# Global mapping for opcodes to operator names
OPCODE_TO_OPNAME = _extract_operator_names()


class Model:
    """A class representing a model that reads and interprets TFLite files.

    Intermediate feature tensors are kept in self.features as a list of lists of numpy tensors:
    self.features[g][n] is the **input* tensor if layer n of subgraph g.
    """


    UNSUPPORTED_FORMAT_MSG = "Format not supported"
    READ_ERROR_MSG = "Error reading TFLite file"
    DECODE_ERROR_MSG = "Error decoding file"

    def __init__(self, filename: str) -> None:
        if not filename.endswith(".tflite"):
            raise ModelError(self.UNSUPPORTED_FORMAT_MSG)

        self.filename = filename
        self.input = None
        self.outputs = []
        self.tfmodel = None
        self.interpreter = None
        self.features = []  # Storing the feature tensors

        # Build TFLite interpreter and parse the model
        self._build_tflite_native_interpreter(filename)
        self._parse_tflite_model(filename)


    def reset_features(self):
        """Initializes all inter-layer features to all zeros."""
        self.features = [] # Clear previous features
        for isg, sg in enumerate(self.tfmodel["subgraphs"]):
            self.features.append([]) # Add an empty list for each subgraph
            for ilayer, layer in enumerate(sg):
                # Create a zero tensor with shape layer ["input_shape"]
                zero_tensor = np.zeros(layer["input_shape"], dtype=np.uint8)
                self.features[isg].append(zero_tensor) # Append zero tensor to features


    def _build_tflite_native_interpreter(self, tflite_model_path: str) -> None:
        """Create tf.lite.Interpreter object out of a .tflite model file."""
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path, experimental_preserve_all_tensors=True)
        self.interpreter.allocate_tensors()

    def _parse_tflite_model(self, filename: str) -> None:
        """Read a TFLite model from a .tflite file and translate it into our internal model format."""
        try:
            with open(filename, "rb") as f:
                buf = f.read()
        
            # Initialize the FlatBuffers Model object
            buf = bytearray(buf)
            tflite_model = TfliteModel.GetRootAs(buf, 0)

            if tflite_model.SubgraphsLength() == 0:
                raise ModelError("No subgraphs found in model.")

            self.tfmodel = {"subgraphs": []}
            
            # Handle multiple subgraphs
            for sg_index in range(tflite_model.SubgraphsLength()):
                subgraph_data = self._translate_subgraph(tflite_model, sg_index)
                self.tfmodel["subgraphs"].append(subgraph_data)

        except Exception as e:
            raise ModelError(f"Error parsing TFLite model: {str(e)}")

    def _translate_subgraph(self, tflite_model, sg_index: int):
        """Translate the subgraph layers into internal format."""
        layers = []
        subgraph = tflite_model.Subgraphs(sg_index)

        for i in range(subgraph.OperatorsLength()):
            operator = subgraph.Operators(i)
            op_index = operator.OpcodeIndex()
            op_code = tflite_model.OperatorCodes(op_index)
            op_name = OPCODE_TO_OPNAME[op_code.BuiltinCode()]

            operator_method = getattr(self, f"_extract_operator_params_{op_name}", None)
            if operator_method is None:
                raise ModelError(f"Unsupported operator '{op_name}' at operator #{i}")
            else:
                layer_data = operator_method(operator, subgraph)
                layers.append(layer_data)

        return layers

    def _get_operator_options(self, operator, OptionsClass):
        """Helper function to extract operator-specific options."""
        options = OptionsClass()
        options.Init(operator.BuiltinOptions().Bytes, operator.BuiltinOptions().Pos)
        return options

    def _extract_operator_params_CONV_2D(self, operator, subgraph):
        """Extract parameters for CONV_2D operator."""
        layer = {"type": "CONV_2D"}
        self._extract_operator_ios(operator, subgraph, layer)
        options = self._get_operator_options(operator, Conv2DOptions)
        layer["stride"] = (options.StrideH(), options.StrideW())
        layer["padding"] = options.Padding()

        # Extract the Conv2D kernel weights
        kernel_tensor_index = operator.InputsAsNumpy()[1]
        kernel_tensor = subgraph.Tensors(kernel_tensor_index)
        kernel_shape = kernel_tensor.ShapeAsNumpy()
        kernel_data = self.interpreter.get_tensor(kernel_tensor_index)
        layer["kernel"] = kernel_data  # Store the kernel (weights)

        return layer

    def _extract_operator_params_MAX_POOL_2D(self, operator, subgraph):
        """Extract parameters for MAX_POOL_2D operator."""
        layer = {"type": "MAX_POOL_2D"}
        self._extract_operator_ios(operator, subgraph, layer)
        options = self._get_operator_options(operator, Pool2DOptions)
        layer["stride"] = (options.StrideH(), options.StrideW())
        layer["padding"] = options.Padding()
        return layer

    def _extract_operator_params_FULLY_CONNECTED(self, operator, subgraph):
        """Extract parameters for FULLY_CONNECTED operator."""
        layer = {"type": "FULLY_CONNECTED"}
        self._extract_operator_ios(operator, subgraph, layer)
        options = self._get_operator_options(operator, FullyConnectedOptions)
        return layer

    def _extract_operator_params_RESHAPE(self, operator, subgraph):
        """Extract parameters for the RESHAPE operator."""
        layer = {"type": "RESHAPE"}
        self._extract_operator_ios(operator, subgraph, layer)
        return layer

    def _extract_operator_params_SOFTMAX(self, operator, subgraph):
        """Extract parameters for the SOFTMAX operator."""
        layer = {"type": "SOFTMAX"}
        self._extract_operator_ios(operator, subgraph, layer)
        return layer

    def _extract_operator_ios(self, operator, subgraph, layer):
        """Extract input/output tensor information for the operator."""
        inputs = operator.InputsAsNumpy()
        outputs = operator.OutputsAsNumpy()
        layer["input_shape"] = self._get_tensor_shape(subgraph, inputs[0])
        layer["output_shape"] = self._get_tensor_shape(subgraph, outputs[0])

        # Store the input and output tensors as feature tensors
        input_tensor = self._get_tensor_data(subgraph, inputs[0])
        output_tensor = self._get_tensor_data(subgraph, outputs[0])

        self.features.append(input_tensor)  # Storing input tensor in features array
        self.features.append(output_tensor)  # Storing output tensor in features array

    def _get_tensor_shape(self, subgraph, tensor_index):
        """Get tensor shape."""
        tensor = subgraph.Tensors(tensor_index)
        return tensor.ShapeAsNumpy()

    def _get_tensor_data(self, subgraph, tensor_index):
        """Get tensor data as numpy array."""
        return self.interpreter.get_tensor(tensor_index)

    def summary(self) -> str:
        """Return a summary of the model."""
        table = PrettyTable()
        table.field_names = ["Operator Index", "Operator Type", "Input Shape", "Output Shape"]
        for i, layer in enumerate(self.tfmodel["subgraphs"][0]):
            table.add_row([i, layer["type"], layer["input_shape"], layer["output_shape"]])
        return table.get_string()

    def layer_summary(self, subgraph: int, layer: int) -> str:
        """Provide a detailed summary of a specific layer in the model."""
        try:
            layer_data = self.tfmodel["subgraphs"][subgraph][layer]
            summary = [f"Layer type: {layer_data['type']}"]
            summary.append(f"Input Shape: {layer_data['input_shape']}")
            summary.append(f"Output Shape: {layer_data['output_shape']}")

            if "stride" in layer_data:
                summary.append(f"Stride: {layer_data['stride']}")
            if "padding" in layer_data:
                summary.append(f"Padding: {layer_data['padding']}")
            if "kernel" in layer_data:
                summary.append(f"Kernel shape: {layer_data['kernel'].shape}")

            return "\n".join(summary)

        except IndexError:
            return f"Error: Layer index {layer} or subgraph index {subgraph} is out of range."

    def set_input_pattern(self, input_pattern: np.ndarray) -> None:
        """Set the input pattern and update the interpreter."""
        input_details = self.interpreter.get_input_details()[0]
        input_index = input_details['index']
        self.interpreter.set_tensor(input_index, input_pattern)
        self.input_tensor = input_pattern

    def _generate_pattern(self, input_shape, pattern, input_type, zero_point):
        """
        Generate an input pattern based on the given shape and pattern type.
        
        Args:
            input_shape (tuple): Shape of the input tensor.
            pattern (int): Pattern type to generate.
            input_type (np.dtype): The data type of the input tensor.
            zero_point (int): Zero point for quantization.

        Returns:
            np.ndarray: Generated input pattern.
        """
        # Create an array with zeros based on the shape and data type
        input_data = np.zeros(input_shape, dtype=input_type)

        # Example pattern generation: fill the input with a certain value
        if pattern == 0:
            input_data.fill(zero_point)
        elif pattern == 1:
            input_data.fill(1)
        elif pattern == 2:
            input_data = np.arange(np.prod(input_shape), dtype=input_type).reshape(input_shape)
        else:
            raise ModelError(f"Unsupported pattern type: {pattern}")

        return input_data

    def run_inference(self) -> np.ndarray:
        # TODO work in progress
        return None
        
    def run_tflite_native_interpreter(self, input: np.ndarray) -> None:
        """Run native TFLite interpreter."""
        input_details = self.interpreter.get_input_details()
        if input.shape != tuple(input_details[0]['shape']):
            raise ModelError(f"Input shape {input.shape} does not match expected shape {tuple(input_details[0]['shape'])}")
        self.interpreter.set_tensor(input_details[0]['index'], input)
        self.interpreter.invoke()

