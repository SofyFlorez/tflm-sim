"""This module contains the definition of the ModelError class and the Model class."""


from pathlib import Path
from prettytable import PrettyTable  # type: ignore[import-not-found]
import numpy as np
import tensorflow as tf
from tflite.Model import Model as TfliteModel
from tflite.BuiltinOperator import BuiltinOperator
import flatbuffers  


class ModelError(Exception):
    """Custom exception for model errors."""

class ModelUserError(Exception):
    """Custom exception for errors caused by bad parameters from user."""

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

    from ._op_conv_2d import _extract_params_CONV_2D
    from ._op_fully_connected import _extract_params_FULLY_CONNECTED
    from ._op_max_pool_2d import _extract_params_MAX_POOL_2D
    from ._op_reshape import _extract_params_RESHAPE
    from ._op_softmax import _extract_params_SOFTMAX


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

            operator_method = getattr(self, f"_extract_params_{op_name}", None)
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

    def _extract_operator_ios(self, operator, subgraph, layer):
        """Extract input/output tensor information for the operator."""
        inputs = operator.InputsAsNumpy()
        outputs = operator.OutputsAsNumpy()
        layer["input_shape"] = self._get_tensor_shape(subgraph, inputs[0])
        layer["output_shape"] = self._get_tensor_shape(subgraph, outputs[0])

        # Remember the in/out tensor TFLite indices because we'll need them to
        # get intermediate layer data from the Interpreter object if we use it
        ilen = operator.InputsLength()
        layer["input_indices"] = [operator.Inputs(x) for x in range(ilen)]
        olen = operator.OutputsLength()
        layer["output_indices"] = [operator.Outputs(x) for x in range(olen)]

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

    def set_input_tensor(self, input: np.ndarray) -> None:
        """Set the input tensor of the model.
        """
        self.input_tensor = input

    def generate_pattern(self, input_shape, pattern, input_type, zero_point=0x80):
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

    def run_inference(self, breakpoint:tuple | None = None) -> np.ndarray:
        """Run the whole model with the given input.

        If the input tensor shape does not match the input shape of the model
        a ModelException will be raised.
        """
        # TODO work in progress
        return None
        
    def run_tflite_native_interpreter(self) -> np.ndarray:
        """Run native TFLite interpreter."""
        input_details = self.interpreter.get_input_details()
        if self.input_tensor.shape != tuple(input_details[0]['shape']):
            raise ModelError(f"Input shape {input.shape} does not match expected shape {tuple(input_details[0]['shape'])}")
        self.interpreter.set_tensor(input_details[0]['index'], self.input_tensor)
        output = self.interpreter.tensor(self.interpreter.get_output_details()[0]["index"])
        self.interpreter.invoke()
        return output()[0]

    def _check_node_indices(self, node:tuple):
        """Check node index tuple (subgraph_index, layer_index) for validity.
        Raise ModelUserError if either index is invalid.
        """
        if len(node) != 2:
            raise ModelUserError("node indices must be 2-tuple of int")
        if node[0] < 0 or node[0] > len(self.tfmodel["subgraphs"]):
            raise ModelUserError("subgraph index out of bounds")
        if node[1] < 0 or node[1] > len(self.tfmodel["subgraphs"][node[0]]):
            raise ModelUserError("layer index out of bounds for subgraph %d" % node[0])

    def get_tflite_native_interpreter_feature(self, node:tuple) -> np.ndarray:
        """Access one of the internal feature tensors of the tflite interpreter
        object.
        This is meant to make it easy to use the interpreter as a golden reference
        for the rest of the package code.
        Returns the selected tensor or raises ModelUserException if the indices
        are wrong.
        """
        self._check_node_indices(node)
        layer = self.tfmodel["subgraphs"][node[0]][node[1]]
        return self.interpreter.get_tensor((layer["output_indices"][0]))

    def get_input_shape(self) -> tuple:
        """Get the shape of the model input.
        """
        return self.tfmodel["subgraphs"][0][0]["input_shape"]

    def get_subgraphs(self) -> list[list[dict]]:
        """Return a list of the model subgraphs.

        The return value is a list of 'subgraphs' (in same order as TFLM file).
        In the context of this function:
        A 'subgraph' is a list of layers  (in execution order).
        Each 'layer' is a dict with some layer attributes as documented in TODO.
        """
        return self.tfmodel["subgraphs"]


    def set_layer_input(self, subgraph_index:int, layer_index:int, tensor: np.ndarray):
        """Set the value of the internal feature tensor at the input of
        layer 'layer_index' of subgraph 'subgraph_index'.

        If the subgraph or layer indices are out of bounds, or the given tensor
        does not have the same shape as the layer input, a ModelException will
        be raised.
        """
        # TODO work in progress
        pass
