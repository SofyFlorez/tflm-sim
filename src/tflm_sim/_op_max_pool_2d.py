
from tflite.Pool2DOptions import Pool2DOptions

def _extract_params_MAX_POOL_2D(self, operator, subgraph):
    """Extract parameters for MAX_POOL_2D operator."""
    layer = {"type": "MAX_POOL_2D"}
    self._extract_operator_ios(operator, subgraph, layer)
    options = self._get_operator_options(operator, Pool2DOptions)
    layer["stride"] = (options.StrideH(), options.StrideW())
    layer["padding"] = options.Padding()
    return layer


