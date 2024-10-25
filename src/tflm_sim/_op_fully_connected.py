
from tflite.FullyConnectedOptions import FullyConnectedOptions

def _extract_params_FULLY_CONNECTED(self, operator, subgraph):
    """Extract parameters for FULLY_CONNECTED operator."""
    layer = {"type": "FULLY_CONNECTED"}
    self._extract_operator_ios(operator, subgraph, layer)
    options = self._get_operator_options(operator, FullyConnectedOptions)
    return layer

