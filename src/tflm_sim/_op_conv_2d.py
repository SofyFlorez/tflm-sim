
from tflite.Conv2DOptions import Conv2DOptions

def _extract_params_CONV_2D(self, operator, subgraph):
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


