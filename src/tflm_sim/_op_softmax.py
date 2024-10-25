

def _extract_params_SOFTMAX(self, operator, subgraph):
    """Extract parameters for the SOFTMAX operator."""
    layer = {"type": "SOFTMAX"}
    self._extract_operator_ios(operator, subgraph, layer)
    return layer


