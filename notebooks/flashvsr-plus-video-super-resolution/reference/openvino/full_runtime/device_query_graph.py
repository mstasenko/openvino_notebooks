"""Bounded layout-only Q outputs for the existing attention-owned GPU slots."""


def expose_device_queries(model, remote_kv=False, *, source_type="f16"):
    """Replace only the original32-window Q Result with two C16 GPU outputs.

    With ``remote_kv=True``, the original K/V Results are also replaced by
    source-order canonical views and one joint finite scalar.  Pooled masks,
    gates, and all arithmetic/storage boundaries remain unchanged.  ``source_type``
    selects the already-materialized typed16bit source storage (``"f16"`` or
    ``"bf16"``); no numeric conversion is inserted.
    """
    import numpy as np
    import openvino as ov
    from openvino import opset13 as ops
    if type(remote_kv) is not bool:
        raise ValueError('remote_kv must be bool')
    if type(source_type) is not str or source_type not in ('f16', 'bf16'):
        raise ValueError('source_type must be f16 or bf16')
    source_dtype = ov.Type.f16 if source_type == 'f16' else ov.Type.bf16
    if source_type == 'f16':
        upper_bound = ops.constant(np.float16(65504), ov.Type.f16)
    else:
        # The largest finite BF16 value is 0x7f7f.  Keep this comparison in
        # BF16; converting the source to F16 would classify finite 65536+ as
        # overflow and would defeat the typed remote boundary.
        bf16_max = np.float32(3.3895314e38)
        upper_bound = ops.constant(bf16_max, ov.Type.bf16)
    names={port.get_any_name() for port in model.outputs}
    if 'q_windows' not in names or names & {'q_device_0','q_device_1','q_finite'}:
        raise ValueError('Unique original Q output required')
    remote_names={'k_device','v_device','kv_finite'}
    if names & remote_names:
        raise ValueError('Unique original K/V outputs required')

    def source(name):
        ports=[port for port in model.outputs if port.get_any_name()==name]
        if len(ports)!=1:
            raise ValueError(f'Unique original {name} output required')
        result=ports[0].get_node()
        if result.get_type_name()!='Result' or result.get_input_size()!=1:
            raise ValueError(f'Original {name} output must be a Result')
        value=result.input_value(0)
        if tuple(value.get_shape())!=(32,128,1536) or value.get_element_type()!=source_dtype:
            raise ValueError(f'Device {name} requires bounded4096-row source pre')
        return result,value

    result,q=source('q_windows')
    kv=[]
    if remote_kv:
        kv=[source(name) for name in ('k_windows','v_windows')]
        if any(value.get_element_type() != q.get_element_type() for _,value in kv):
            raise ValueError('Device Q/K/V require one consistent typed16bit storage')
    layout=ops.transpose(ops.reshape(q,ops.constant([32,128,12,128],ov.Type.i64),False),
                         ops.constant([2,0,1,3],ov.Type.i64))
    split=ops.split(layout,ops.constant(1,ov.Type.i64),2)
    # Ordered <= rejects NaNs as well as infinities. Abs removes the second
    # comparison/AND without changing the inclusive finite storage bounds.
    finite=ops.less_equal(ops.absolute(q),upper_bound)
    valid=ops.reduce_min(ops.convert(finite,ov.Type.f16),ops.constant([0,1,2],ov.Type.i64),False)
    outputs=[*split.outputs(),valid.output(0)]
    if remote_kv:
        canonical=[]
        for (name,(source_result,value)) in zip(('k_device','v_device'),kv):
            view=ops.reshape(value,ops.constant([32,128,12,128],ov.Type.i64),False)
            view.output(0).get_tensor().set_names({name})
            canonical.append(view.output(0))
        k,v=kv[0][1],kv[1][1]
        k_finite=ops.less_equal(ops.absolute(k),upper_bound)
        v_finite=ops.less_equal(ops.absolute(v),upper_bound)
        kv_valid=ops.reduce_min(
            ops.convert(ops.logical_and(k_finite,v_finite),ov.Type.f16),
            ops.constant([0,1,2],ov.Type.i64),False)
        kv_valid.output(0).get_tensor().set_names({'kv_finite'})
        outputs.extend([*canonical,kv_valid.output(0)])
    for port,name in zip(outputs,('q_device_0','q_device_1','q_finite')):
        port.get_tensor().set_names({name})
    for old_result,_value in [(result,q),*kv]:
        model.remove_result(old_result)
    model.add_outputs(outputs)
    model.validate_nodes_and_infer_types()

