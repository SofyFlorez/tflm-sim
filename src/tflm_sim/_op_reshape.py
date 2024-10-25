

def _extract_params_RESHAPE(self, operator, subgraph):
    """Extract parameters for the RESHAPE operator."""
    layer = {"type": "RESHAPE"}
    self._extract_operator_ios(operator, subgraph, layer)
    return layer


