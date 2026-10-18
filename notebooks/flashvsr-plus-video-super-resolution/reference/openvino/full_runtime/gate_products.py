"""Observe source F16 projection/product boundaries; cloned lab graphs only.

GPU post-op fusion otherwise keeps the product at higher precision until the
residual add. Results make the two source storage boundaries observable. This
is an accuracy experiment, not a selected production path or a speed claim.
"""
from math import prod


def preserve(model):
    import openvino as ov
    if len(model.outputs) != 1 or model.output(0).get_any_name() != "x_out":
        raise ValueError("Expected final-only post graph")
    if tuple(model.input("x").get_shape()) not in ((1,23040,1536),(1,7680,1536),(1,195840,1536),(1,65280,1536)):
        raise ValueError("Only reviewed HD/UHD post metadata is admitted")
    inputs = [(p.get_any_name(),tuple(p.get_shape()),str(p.get_element_type())) for p in model.inputs]
    nodes = model.get_ordered_ops()
    parameters = {p.output(0).get_any_name():p for p in model.get_parameters()}
    observed = []
    for projection,gate_name in (("self","gate_msa"),("ffn","gate_mlp")):
        found = [node for node in nodes if node.get_friendly_name() == f"token_batch_{projection}_output_reshape"]
        if len(found) != 1 or gate_name not in parameters:
            raise ValueError("Requires authenticated token-batch projection and gate")
        value = found[0].output(0)
        consumers = [target.get_node() for target in value.get_target_inputs()]
        if len(consumers) != 1 or consumers[0].get_type_name() != "Multiply":
            raise ValueError("Projection must have one gate multiply")
        product = consumers[0]
        sources = [product.input_value(i).get_node() for i in (0,1)]
        if parameters[gate_name] not in sources or found[0] not in sources:
            raise ValueError("Gate multiply inputs changed")
        for suffix,port in (("projected",value),("product",product.output(0))):
            if port.get_element_type() != ov.Type.f16 or port.get_partial_shape() != model.input("x").get_partial_shape():
                raise ValueError("Source gate storage ABI changed")
            observed.append((f"gate_{projection}_{suffix}",port))
    # Authenticate both gates before mutating the caller-owned clone.
    for name,port in observed:
        port.get_tensor().set_names({name})
        model.add_outputs([port])
    model.validate_nodes_and_infer_types()
    if inputs != [(p.get_any_name(),tuple(p.get_shape()),str(p.get_element_type())) for p in model.inputs]:
        raise ValueError("Gate observation changed input ABI")
    return {"storage":"observable F16 Results","extra_outputs":[name for name,_ in observed]}


def bind_remote_results(compiled,request,context):
    """Retain storage barriers without copying their unused values to host."""
    import openvino as ov
    expected = {f"gate_{projection}_{suffix}" for projection in ("self","ffn")
                for suffix in ("projected","product")}
    ports = {port.get_any_name():port for port in compiled.outputs}
    if len(compiled.outputs) != 5 or set(ports) != {"x_out",*expected}:
        raise ValueError("Remote gate Results require the exact observed post ABI")
    if ports["x_out"].get_element_type() != ov.Type.f16:
        raise ValueError("Remote gate output must preserve F16 state")
    shape = tuple(compiled.output("x_out").get_shape())
    for name in expected:
        port = ports[name]
        if port.get_element_type() != ov.Type.f16 or tuple(port.get_shape()) != shape:
            raise ValueError("Remote gate Result storage differs from x_out")
    for name in sorted(expected):
        port = ports[name]
        tensor = context.create_tensor(ov.Type.f16,port.get_shape(),{})
        request.set_tensor(name,tensor)
        # The request owns its bound Tensor; do not retain a separate startup
        # bank that would survive retirement of the f6 request.
        del tensor
    return {"remote_outputs":sorted(expected),"bytes":len(expected)*prod(shape)*2}
