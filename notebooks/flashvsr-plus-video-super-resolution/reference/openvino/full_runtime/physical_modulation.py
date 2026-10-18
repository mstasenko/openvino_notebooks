"""GPU-owned pre/post experiments with original F16 storage boundaries."""
import numpy as np
from half_storage import all_finite


def split(model,role="post",gates=False,f32_gelu=False,inline_gelu=False,ffn_rows=None,linear_store=False,norm_f32=False,qk_store=False,cross_stores=False,rms_mode=None):
    """Split a caller-owned cloned graph at normalization/product, not an oracle."""
    import openvino as ov
    from openvino import opset13 as ops
    if rms_mode is not None and (rms_mode not in ('math','stores','both') or role != 'pre' or not qk_store):
        raise ValueError('RMS control requires original pre Q/K stores and an explicit math/stores variant')
    params = {p.output(0).get_any_name():p for p in model.get_parameters()}
    shape = tuple(model.input("x").get_shape())
    bounded_pre=(role=="pre" and len(shape)==3 and shape[0]==1 and shape[2]==1536
                 and 128<=shape[1]<=4096 and shape[1]%128==0)
    uhd_post=role=="post" and shape in ((1,195840,1536),(1,65280,1536))
    if role not in ("pre","post") or (shape not in ((1,23040,1536),(1,7680,1536)) and not bounded_pre and not uhd_post):
        raise ValueError("Physical modulation requires reviewed HD pre/post")
    if (norm_f32 or qk_store) and role != "pre":
        raise ValueError("F32 normalization control is restricted to the small pre front")
    if role == "post" and (len(model.outputs) != 1 or model.output(0).get_any_name() != "x_out"):
        raise ValueError("Physical modulation requires final-only post")
    if gates and role != "post":
        raise ValueError("Gate product storage requires the reviewed post graph")
    if cross_stores and role != "post":
        raise ValueError("Cross projection stores require the reviewed post graph")
    if (f32_gelu or inline_gelu) and (role != "post" or (f32_gelu and inline_gelu)):
        raise ValueError("F32 GELU requires the reviewed post graph")
    if linear_store and not inline_gelu:
        raise ValueError("Linear Result storage requires inline GELU")
    if ffn_rows is not None and (type(ffn_rows)!=int or ffn_rows not in (1280,3840,7680)
            or shape[1]%ffn_rows or not (f32_gelu or inline_gelu) or gates):
        raise ValueError("Bounded FFN requires reviewed exact rows and F32 GELU without extra gate outputs")
    weight = params["weight__blocks_0_ffn_0_weight" if role == "post" else "weight__blocks_0_self_attn_q_weight"]
    consumers = [target.get_node() for target in weight.output(0).get_target_inputs() if target.get_index() == 1]
    if len(consumers) != 1 or consumers[0].get_type_name() != "MatMul":
        raise ValueError("Modulated input projection changed")
    added = consumers[0].input_value(0).get_node()
    if added.get_type_name() != "Add":
        raise ValueError("Expected source modulation shift add")
    if role == "post" and added.input_value(1).get_node() != params["shift_mlp"]:
        raise ValueError("Post shift Parameter changed")
    product = added.input_value(0).get_node()
    if product.get_type_name() != "Multiply":
        raise ValueError("Expected modulation product")
    normed,gain = (product.input_value(i).get_node() for i in (0,1))
    if (normed.get_type_name() != "Convert" or normed.input_value(0).get_element_type() != ov.Type.f32
            or gain.get_type_name() != "Add"):
        raise ValueError("Expected source norm2/gain boundaries")
    constant = gain.input_value(0).get_node()
    if constant.get_type_name() != "Constant" or list(constant.get_vector()) != [1.0]:
        raise ValueError("Expected exact one+scale gain")
    if role == "post":
        if gain.input_value(1).get_node() != params["scale_mlp"]:
            raise ValueError("Post scale Parameter changed")
    else:
        def chunk(node,index):
            if node.get_type_name() != "StridedSlice": raise ValueError("Pre modulation slice changed")
            for input_index,expected in ((1,[0,index,0]),(2,[1,index+1,1536]),(3,[1,1,1])):
                value=node.input_value(input_index).get_node()
                if value.get_type_name() != "Constant" or list(value.get_vector()) != expected:
                    raise ValueError("Pre modulation slice bounds changed")
            return node.input_value(0).get_node()
        origin = chunk(gain.input_value(1).get_node(),1)
        if origin != chunk(added.input_value(1).get_node(),0) or origin.get_type_name() != "Add":
            raise ValueError("Pre shift/scale must share the original modulation")
        if {origin.input_value(i).get_node() for i in (0,1)} != {
                params["t_mod"],params["weight__blocks_0_modulation"]}:
            raise ValueError("Pre modulation Parameters changed")
    # The reviewed non-affine norm is Convert(Divide(Subtract(Convert(x),mean),std)).
    divide = normed.input_value(0).get_node()
    if divide.get_type_name() != "Divide": raise ValueError("Norm2 division changed")
    centered = divide.input_value(0).get_node()
    if centered.get_type_name() != "Subtract": raise ValueError("Norm2 centering changed")
    converted = centered.input_value(0).get_node()
    if converted.get_type_name() != "Convert": raise ValueError("Norm2 source conversion changed")
    residual = converted.input_value(0).get_node()
    for name,node,expected in (("normed",normed,shape),("residual",residual,shape),("gain",gain,(1,1,1536))):
        if node.output(0).get_element_type() != ov.Type.f16 or tuple(node.output(0).get_shape()) != expected:
            raise ValueError(f"Source {name} F16 storage changed")

    def parameters(outputs,allowed):
        reached = set()
        pending = [port.get_node() for port in outputs]
        while pending:
            node = pending.pop()
            if node.get_instance_id() in reached: continue
            reached.add(node.get_instance_id())
            pending.extend(value.get_node() for value in node.input_values())
        return [p for p in allowed if p.get_instance_id() in reached]

    boundaries = [("modulation_normed",normed),("modulation_gain",gain)]
    if role == "post": boundaries.append(("modulation_residual",residual))
    if gates:
        from gate_products import preserve
        preserve(model)
        for name in ("gate_self_projected","gate_self_product"):
            boundaries.append((name,model.output(name).get_node().input_value(0).get_node()))
    if cross_stores:
        for lane in ("q","o"):
            parameter=params[f"weight__blocks_0_cross_attn_{lane}_weight"]
            products=[t.get_node()for t in parameter.output(0).get_target_inputs()if t.get_index()==1]
            if len(products)!=1 or products[0].get_type_name()!="MatMul":
                raise ValueError("Original cross projection MatMul changed")
            users=[t.get_node()for t in products[0].output(0).get_target_inputs()]
            if (len(users)!=1 or users[0].get_type_name()!="Add"
                    or users[0].output(0).get_element_type()!=ov.Type.f16
                    or tuple(users[0].output(0).get_shape())!=shape):
                raise ValueError("Original half cross projection bias/store changed")
            boundaries.append((f"modulation_cross_{lane}_linear",users[0]))
    for name,node in boundaries:
        node.output(0).get_tensor().set_names({name})
    outputs = [node.output(0) for _,node in boundaries]
    # Clone before cutting the tail: otherwise replace() also rewires the front.
    front = ov.Model(outputs,parameters(outputs,model.get_parameters()),"modulation_front").clone()
    stored = ops.parameter(shape,ov.Type.f16,name="modulation_scaled")
    old = ops.parameter(shape,ov.Type.f16,name="modulation_residual")
    product.output(0).replace(stored.output(0))
    if role == "post": residual.output(0).replace(old.output(0))
    outputs = [port.get_node().input_value(0) for port in model.outputs
               if not (gates and port.get_any_name().startswith("gate_self_"))]
    tail = ov.Model(outputs,parameters(outputs,[*model.get_parameters(),stored,old]),"modulation_tail")
    if qk_store:
        observed=[]
        for lane in ("q","k"):
            weight=params[f"weight__blocks_0_self_attn_{lane}_weight"]
            targets=[t.get_node() for t in weight.output(0).get_target_inputs() if t.get_index()==1]
            if len(targets)!=1 or targets[0].get_type_name()!="MatMul":
                raise ValueError("Original pre Q/K projection changed")
            users=[t.get_node() for t in targets[0].output(0).get_target_inputs()]
            if len(users)!=1 or users[0].get_type_name()!="Add":
                raise ValueError("Original pre Q/K bias Add changed")
            value=users[0].output(0)
            if value.get_element_type()!=ov.Type.f16 or tuple(value.get_shape())!=shape:
                raise ValueError("Original pre Q/K storage ABI changed")
            observed.append(("modulation_projected_"+lane,value))
            if rms_mode is not None:
                scale = params[f'weight__blocks_0_self_attn_norm_{lane}_weight']
                reshape, = [t.get_node() for t in scale.output(0).get_target_inputs()]
                if reshape.get_type_name() != 'Reshape':
                    raise ValueError('RMS learned scale reshape changed')
                weighted, = [t.get_node() for t in reshape.output(0).get_target_inputs()]
                normalized = weighted.input_value(0).get_node()
                if (weighted.get_type_name() != 'Multiply' or weighted.input_value(1).get_node() != reshape
                        or normalized.get_type_name() != 'Convert' or normalized.output(0).get_element_type() != ov.Type.f16):
                    raise ValueError('Original RMS half normalization/scale boundaries changed')
                divide = normalized.input_value(0).get_node()
                if (divide.get_type_name() != 'Divide' or divide.output(0).get_element_type() != ov.Type.f32
                        or divide.input_value(1).get_node().get_type_name() != 'Sqrt'):
                    raise ValueError('Original RMS FP32 divide/sqrt changed')
                if rms_mode in ('math', 'both'):
                    variance = divide.input_value(1).get_node().input_value(0)
                    inverse = ops.power(variance, ops.constant(np.float32(-.5)))
                    divide.output(0).replace(ops.multiply(divide.input_value(0), inverse).output(0))
                if rms_mode in ('stores', 'both'):
                    # Keep both original half stores visible to the GPU compiler;
                    # these Results remain GPU-only like the Linear boundaries.
                    observed.extend((f'modulation_projected_rms_{name}_{lane}', node.output(0))
                                    for name, node in (('normalized', normalized), ('weighted', weighted)))
        # These GPU-only Results prevent FP32 FC output from bypassing the
        # original half Linear store before RMSNorm. No new request needed.
        for name,value in observed:
            value.get_tensor().set_names({name})
            tail.add_outputs([value])
    normalized = ops.parameter(shape,ov.Type.f16,name="modulation_normed")
    scale = ops.parameter([1,1,1536],ov.Type.f16,name="modulation_gain")
    scaled = ops.multiply(normalized,scale)
    scaled.output(0).get_tensor().set_names({"modulation_scaled"})
    multiply = ov.Model([scaled],[normalized,scale],"modulation_multiply")
    graphs=[front,multiply,tail]
    if norm_f32:
        # Keep gain/modulation Add operations on their original F16 path.
        # A F32 hint on the whole front would also promote those operations.
        normal = front.output("modulation_normed").get_node().input_value(0)
        gain_output = front.output("modulation_gain").get_node().input_value(0)
        normal_graph = ov.Model([normal],parameters([normal],front.get_parameters()),"source_pre_norm_opmath")
        gain_graph = ov.Model([gain_output],parameters([gain_output],front.get_parameters()),"source_pre_gain_f16")
        graphs=[normal_graph,gain_graph,multiply,tail]
    if f32_gelu or inline_gelu:
        gelu,=[node for node in tail.get_ordered_ops()if node.get_type_name()=="Gelu"]
        if (str(gelu.get_attributes()["approximation_mode"]).lower()!="tanh"
                or gelu.input_value(0).get_element_type()!=ov.Type.f16
                or tuple(gelu.input_value(0).get_shape())!=(1,shape[1],8960)):
            raise ValueError("Source FFN tanh-GELU ABI changed")
    if inline_gelu:
        if linear_store:
            linear=gelu.input_value(0)
            linear.get_tensor().set_names({"modulation_ffn_linear"})
            tail.add_outputs([linear])
        result=ops.convert(ops.gelu(ops.convert(gelu.input_value(0),ov.Type.f32),"tanh"),ov.Type.f16)
        result.output(0).get_tensor().set_names(gelu.output(0).get_names())
        gelu.output(0).replace(result.output(0))
        tail.set_friendly_name("source_ffn_gelu_inline")
    if f32_gelu:
        linear=gelu.input_value(0)
        linear.get_tensor().set_names({"modulation_ffn_linear"})
        projection=ov.Model([linear],parameters([linear],tail.get_parameters()),"source_ffn_linear").clone()
        hidden=ops.parameter([1,shape[1],8960],ov.Type.f16,name="modulation_ffn_hidden")
        gelu.output(0).replace(hidden.output(0))
        outputs=[port.get_node().input_value(0)for port in tail.outputs]
        final=ov.Model(outputs,parameters(outputs,[*tail.get_parameters(),hidden]),"source_ffn_final")
        x=ops.parameter([1,shape[1],8960],ov.Type.f16,name="modulation_ffn_linear")
        y=ops.convert(ops.gelu(ops.convert(x,ov.Type.f32),"tanh"),ov.Type.f16)
        y.output(0).get_tensor().set_names({"modulation_ffn_hidden"})
        activation=ov.Model([y],[x],"source_ffn_gelu_opmath")
        graphs=[front,multiply,projection,activation,final]
    if ffn_rows is not None:
        # FFN is pointwise in token rows. Keep the original token-batch
        # Conv layout, changing only its two authenticated shape literals.
        shapes=[]
        for graph in graphs[2:]:
            for node in graph.get_ordered_ops():
                expected={"token_batch_ffn_input_reshape":[shape[1],8960,1,1],
                          "token_batch_ffn_output_reshape":[1,shape[1],1536]}.get(node.get_friendly_name())
                if expected is not None:
                    value=node.input_value(1).get_node()
                    if value.get_type_name()!="Constant" or list(value.get_vector())!=expected:
                        raise ValueError("Original FFN token-batch shape changed")
                    resized=list(expected); resized[0 if len(resized)==4 else 1]=ffn_rows
                    shapes.append((node,resized))
        if len(shapes)!=2: raise ValueError("Expected both reviewed FFN token-batch reshapes")
        for node,resized in shapes:
            node.input(1).replace_source_output(ops.constant(resized,ov.Type.i64).output(0))
        for graph in graphs[2:]:
            for param in graph.get_parameters():
                if param.output(0).get_any_name().startswith("modulation_"):
                    dimensions=list(param.output(0).get_shape()); dimensions[1]=ffn_rows
                    param.set_partial_shape(ov.PartialShape(dimensions))
    for graph in graphs: graph.validate_nodes_and_infer_types()
    return tuple(graphs)


def block_post_graphs(graphs, tokens, rows):
    """Resize only authenticated token axes; cross-attention K/V stay whole."""
    import openvino as ov
    from openvino import opset13 as ops
    if tokens not in (23040,7680,195840,65280) or type(rows)!=int or rows not in (1280,3840,7680) or tokens%rows:
        raise ValueError("Whole post blocks require reviewed exact token rows")
    for graph in graphs:
        for node in graph.get_ordered_ops():
            if node.get_type_name() == "Constant" and node.output(0).get_element_type()==ov.Type.i64:
                values=np.asarray(node.get_data())
                if not np.any(values==tokens):
                    continue
                uses=list(node.output(0).get_target_inputs())
                if (values.ndim!=1 or not uses or any(t.get_index()!=1 or
                        t.get_node().get_type_name() not in ("Reshape","Broadcast") for t in uses)):
                    raise ValueError("Token literal is not an authenticated reshape/broadcast dimension")
                resized=np.where(values==tokens,rows,values)
                node.output(0).replace(ops.constant(resized,ov.Type.i64).output(0))
        for param in graph.get_parameters():
            dims=list(param.output(0).get_shape())
            if tokens in dims:
                name=param.output(0).get_any_name()
                if (len(dims)!=3 or dims[:2]!=[1,tokens] or
                        not (name in ("x","self_context") or name.startswith("modulation_"))):
                    raise ValueError("Unexpected token-dependent post Parameter")
                dims[1]=rows
                param.set_partial_shape(ov.PartialShape(dims))
        graph.validate_nodes_and_infer_types()
    return graphs


class Modulation:
    """GPU-owned F16 boundaries; read back only the original public outputs."""
    def __init__(self,core,device,model,config,role="post",gates=False,f32_gelu=False,inline_gelu=False,ffn_rows=None,inline_accuracy=False,linear_store=False,norm_f32=False,qk_store=False,tail_accuracy=False,post_accuracy=False,post_rows=None,cross_stores=False,rms_mode=None,flat_stage_outputs=False,remote_kv=False,reuse_host_output=False):
        import openvino as ov
        if type(reuse_host_output) is not bool or (reuse_host_output and (role!='post' or post_rows is None)):
            raise ValueError('Output reuse requires bounded synchronous post')
        self.reuse_host_output = reuse_host_output
        self._host_output = None
        if type(remote_kv) is not bool:
            raise ValueError("remote_kv must be bool")
        model_output_names={port.get_any_name() for port in model.outputs}
        device_names={name for name in model_output_names if name.startswith("q_device_")}
        q_remote_names=model_output_names & {"q_device_0","q_device_1"}
        kv_remote_names=model_output_names & {"k_device","v_device"}
        remote_names=q_remote_names | kv_remote_names
        if device_names and device_names!={"q_device_0","q_device_1"}:
            raise ValueError("Remote Q output ABI changed")
        if q_remote_names and (role!="pre" or q_remote_names!={"q_device_0","q_device_1"}
                             or "q_finite" not in model_output_names):
            raise ValueError("Remote Q outputs require the complete pre graph")
        if "q_finite" in model_output_names and not q_remote_names:
            raise ValueError("q_finite requires remote Q outputs")
        if remote_kv and (role != "pre" or q_remote_names != {"q_device_0","q_device_1"}
                          or kv_remote_names != {"k_device","v_device"}
                          or "kv_finite" not in model_output_names):
            raise ValueError("Remote K/V outputs require the complete pre graph")
        if not remote_kv and (kv_remote_names or "kv_finite" in model_output_names):
            raise ValueError("Remote K/V outputs require remote_kv=True")
        if remote_kv:
            remote_names |= {"k_device","v_device"}
        if q_remote_names:
            for name in ("q_device_0","q_device_1"):
                port=model.output(name)
                if (port.get_element_type()!=ov.Type.f16 or
                        tuple(port.get_shape())!=(12,16,128,128)):
                    raise ValueError("Remote Q output ABI changed")
            finite=model.output("q_finite")
            if finite.get_element_type()!=ov.Type.f16 or tuple(finite.get_shape())!=():
                raise ValueError("Remote Q finite output ABI changed")
        if remote_kv:
            for name in ("k_device","v_device"):
                port=model.output(name)
                if (port.get_element_type()!=ov.Type.f16 or
                        tuple(port.get_shape())!=(32,128,12,128)):
                    raise ValueError("Remote K/V output ABI changed")
            finite=model.output("kv_finite")
            if finite.get_element_type()!=ov.Type.f16 or tuple(finite.get_shape())!=():
                raise ValueError("Remote K/V finite output ABI changed")
        self.remote_output_names=frozenset(remote_names)
        self.remote_kv=remote_kv
        if inline_accuracy and not inline_gelu:
            raise ValueError("Inline accuracy requires the explicit F32 GELU graph")
        if rms_mode is not None and (role != 'pre' or not qk_store or not tail_accuracy
                                     or rms_mode not in ('math','stores','both')):
            raise ValueError('RMS control requires source-precision pre tail')
        if (norm_f32 or qk_store) and role != "pre":
            raise ValueError("F32 normalization control is restricted to the small pre front")
        if tail_accuracy and (role != "pre" or not qk_store):
            raise ValueError("Pre tail accuracy requires original half Q/K projection stores")
        if post_accuracy and role != "post":
            raise ValueError("Post precision control requires the post front")
        if post_rows is not None and (role!="post" or ffn_rows is not None):
            raise ValueError("Whole post blocking cannot combine with pre or FFN-only blocking")
        if flat_stage_outputs and not ((role == 'pre' and 128 <= int(model.input('x').get_shape()[1]) <= 4096)
                                       or (role == 'post' and post_rows is not None)):
            raise ValueError('Flat output views require bounded pre/post stages')
        shape=tuple(model.input("x").get_shape()) if role=="post" else None
        # A UHD post graph is metadata only until every request is bounded.
        # Reject before even creating a GPU context; never compile it whole.
        if role=="post" and shape in ((1,195840,1536),(1,65280,1536)) and (
                type(post_rows)!=int or post_rows not in (1280,3840)):
            raise ValueError("UHD post requires explicit bounded query rows")
        self.post_rows=post_rows
        self.full_shape=tuple(model.input("x").get_shape()) if post_rows is not None else None
        self.context = core.get_default_context(device)
        self.engines = []
        self.role = role
        graphs=split(model,role,gates,f32_gelu,inline_gelu,ffn_rows,linear_store,norm_f32,qk_store,cross_stores,rms_mode)
        if post_rows is not None:
            graphs=block_post_graphs(graphs,self.full_shape[1],post_rows)
        layouts = {}
        if flat_stage_outputs:
            from flat_output_views import flatten_results
            layouts = flatten_results(graphs,internal_only=role=='pre')
        for graph in graphs:
            # The normal F16 hint lowers even explicitly F32 GELU. Only this
            # activation request needs source FP32 opmath; weights stay F16.
            options=({**config,"INFERENCE_PRECISION_HINT":ov.Type.f32}
                     if graph.get_friendly_name() in ("source_ffn_gelu_opmath","source_ffn_gelu_inline") else config)
            if norm_f32 and graph.get_friendly_name() == "source_pre_norm_opmath":
                options={**config,"INFERENCE_PRECISION_HINT":ov.Type.f32}
            if tail_accuracy and graph.get_friendly_name() == "modulation_tail":
                # Honor the existing F32 RMS/RoPE ops, not a global FP32
                # projection hint. The learned projection ports remain F16.
                options=dict(config)
                options.pop("INFERENCE_PRECISION_HINT",None)
                options["EXECUTION_MODE_HINT"]=ov.properties.hint.ExecutionMode.ACCURACY
            if post_accuracy and graph.get_friendly_name() == "modulation_front":
                options=dict(config)
                options.pop("INFERENCE_PRECISION_HINT",None)
                options["EXECUTION_MODE_HINT"]=ov.properties.hint.ExecutionMode.ACCURACY
            if inline_accuracy and graph.get_friendly_name()=="source_ffn_gelu_inline":
                options=dict(config)
                options.pop("INFERENCE_PRECISION_HINT",None)
                options["EXECUTION_MODE_HINT"]=ov.properties.hint.ExecutionMode.ACCURACY
            compiled = core.compile_model(graph,self.context,options)
            if list(compiled.get_property("EXECUTION_DEVICES")) != [device]:
                raise ValueError("Physical modulation used another device")
            self.engines.append((compiled,compiled.create_infer_request()))
        tail=self.engines[-1]
        buffers = {}
        self._output_storages = []
        for compiled,request in self.engines:
            for port in compiled.outputs:
                name = port.get_any_name()
                if name in self.remote_output_names:
                    continue
                if request is tail[1] and not ((gates and name.startswith("gate_ffn_"))
                                               or (linear_store and name=="modulation_ffn_linear")
                                               or (qk_store and name.startswith("modulation_projected_"))):
                    continue
                buffer = self.context.create_tensor(port.get_element_type(),port.get_shape(),{})
                request.set_tensor(name,buffer)
                self._output_storages.append(buffer)
                buffers[name] = (self.context.create_tensor(port.get_element_type(),ov.Shape(layouts[name]),buffer.get_params())
                                 if name in layouts else buffer)
        for compiled,request in self.engines:
            for port in compiled.inputs:
                name=port.get_any_name()
                if name.startswith("modulation_"):
                    if tuple(port.get_shape())==tuple(buffers[name].get_shape()):
                        request.set_tensor(name,buffers[name])
                    elif ffn_rows is None or name not in ("modulation_scaled","modulation_residual"):
                        raise ValueError("Internal GPU boundary shape changed")
        self.blocks=[]
        self.post_blocks=[]
        self.output=None
        if post_rows is not None:
            shape = self.full_shape[1:] if flat_stage_outputs else self.full_shape
            self._post_output_storage=self.context.create_tensor(ov.Type.f16,ov.Shape(shape),{})
            self.output=(self.context.create_tensor(ov.Type.f16,ov.Shape(self.full_shape),self._post_output_storage.get_params())
                         if flat_stage_outputs else self._post_output_storage)
            buffers["x_out"]=self.output
            for start in range(0,self.full_shape[1],post_rows):
                begin,end=([start,0],[start+post_rows,self.full_shape[-1]]) if flat_stage_outputs else (
                    [0,start,0],[1,start+post_rows,self.full_shape[-1]])
                view=ov.RemoteTensor(self._post_output_storage,ov.Coordinate(begin),ov.Coordinate(end))
                self.post_blocks.append((start,view))
        if ffn_rows is not None:
            shape=model.input("x").get_shape()
            self.output=self.context.create_tensor(ov.Type.f16,shape,{})
            buffers["x_out"]=self.output
            for start in range(0,shape[1],ffn_rows):
                views=[]
                final_index=len(self.engines)-1
                for index,name in ((2,"modulation_scaled"),(final_index,"modulation_residual"),(final_index,"x_out")):
                    width=buffers[name].get_shape()[-1]
                    view=ov.RemoteTensor(buffers[name],ov.Coordinate([0,start,0]),
                                         ov.Coordinate([1,start+ffn_rows,width]))
                    views.append((index,name,view))
                self.blocks.append(views)
        self.bytes = sum(buffer.get_byte_size() for buffer in buffers.values())
        self.feed_names = {port.get_any_name() for compiled,_ in self.engines for port in compiled.inputs
                           if not port.get_any_name().startswith(("weight__","modulation_"))}
        self.weight_names = {port.get_any_name() for compiled,_ in self.engines for port in compiled.inputs
                             if port.get_any_name().startswith("weight__")}

    def _validate_remote_outputs(self,remote_outputs):
        import openvino as ov
        expected=getattr(self,"remote_output_names",frozenset())
        if remote_outputs is None:
            supplied=None
        elif isinstance(remote_outputs,dict):
            supplied=set(remote_outputs)
        else:
            supplied=object()
        if not expected:
            if remote_outputs is not None:
                raise ValueError("Remote outputs are valid only for the remote pre graph")
            return {}
        if supplied!=expected or not isinstance(remote_outputs,dict):
            fields=", ".join(sorted(expected))
            raise ValueError(f"Remote pre requires exactly {fields} outputs")
        remote_type=getattr(ov,"RemoteTensor",None)
        if remote_type is None:
            raise ValueError("OpenVINO RemoteTensor support is required for remote Q")
        compiled,request=self.engines[-1]
        ports={port.get_any_name():port for port in compiled.outputs}
        for name in expected:
            port=ports.get(name);value=remote_outputs[name]
            expected_shape=(12,16,128,128) if name.startswith("q_device_") else (32,128,12,128)
            if (port is None or port.get_element_type()!=ov.Type.f16 or
                    tuple(port.get_shape())!=expected_shape):
                raise ValueError("Remote pre output port ABI changed")
            try:
                actual_shape=tuple(value.get_shape())
                byte_size=int(value.get_byte_size())
            except Exception as error:
                raise ValueError("Remote output must be a real RemoteTensor") from error
            expected_bytes=int(np.prod(expected_shape))*2
            if (not isinstance(value,remote_type) or actual_shape!=expected_shape
                    or byte_size!=expected_bytes):
                label="Q" if name.startswith("q_device_") else "K/V"
                raise ValueError(f"Remote {label} output must be an F16 RemoteTensor")
        return remote_outputs

    def infer(self,feeds,weights,remote_outputs=None):
        import openvino as ov
        remote_outputs=self._validate_remote_outputs(remote_outputs)
        if set(feeds) != self.feed_names or not self.weight_names <= set(weights):
            raise ValueError("Physical modulation feed/weight fields changed")
        if getattr(self,"post_rows",None) is not None:
            for name in ("x","self_context"):
                value = feeds[name]
                remote = isinstance(value, ov.RemoteTensor)
                valid = ((name == 'self_context' and tuple(value.get_shape()) == self.full_shape
                          and value.get_byte_size() == int(np.prod(self.full_shape)) * 2) if remote
                         else (value.shape == self.full_shape and value.dtype == np.float16))
                if not valid:
                    raise ValueError("Whole post blocking requires original complete F16 inputs")
        # Bind/check every stage before starting any GPU work. Internal inputs
        # stay bound to this object's own F16 buffers, never caller overrides.
        host_inputs = []
        for compiled,request in self.engines:
            host = {}
            for port in compiled.inputs:
                name = port.get_any_name()
                if name.startswith("weight__"):
                    request.set_tensor(name,weights[name])
                elif name in feeds:
                    if not isinstance(feeds[name], ov.RemoteTensor):
                        host[name] = feeds[name]
                    elif name != 'self_context' or not getattr(self, 'post_rows', None):
                        raise ValueError('Only bounded self-context may be a caller-owned remote tensor')
            host_inputs.append(host)
        for name,value in remote_outputs.items():
            self.engines[-1][1].set_tensor(name,value)
        post_blocks=getattr(self,"post_blocks",[])
        # Every submission is waited here before input reuse; caller arrays
        # stay alive and unchanged. Borrow them instead of a second host copy.
        for start,view in post_blocks:
            self.engines[-1][1].set_tensor("x_out",view)
            for (compiled,request),host in zip(self.engines,host_inputs):
                if (isinstance(feeds.get('self_context'), ov.RemoteTensor)
                        and any(p.get_any_name() == 'self_context' for p in compiled.inputs)):
                    view = ov.RemoteTensor(feeds['self_context'], ov.Coordinate([0,start,0]),
                                           ov.Coordinate([1,start+self.post_rows,self.full_shape[-1]]))
                    request.set_tensor('self_context', view)
                block={name:(value[:,start:start+self.post_rows] if name in ("x","self_context") else value)
                       for name,value in host.items()}
                request.start_async(block or None,share_inputs=True)
                request.wait()
        for (_,request),host in zip([] if post_blocks else (self.engines[:2] if self.blocks else self.engines),host_inputs):
            request.start_async(host or None,share_inputs=True)
            request.wait()
        if self.blocks:
            for views in self.blocks:
                for index,name,view in views:
                    self.engines[index][1].set_tensor(name,view)
                for index in range(2,len(self.engines)):
                    request=self.engines[index][1]
                    request.start_async(host_inputs[index] or None,share_inputs=True)
                    request.wait()
        if self.blocks or post_blocks:
            # Copy GPU output directly into the returned NumPy-owned storage.
            # A separate Tensor allocation plus np.array duplicates up to600MB.
            value=getattr(self,'_host_output',None) if getattr(self,'reuse_host_output',False) else None
            if value is None:
                value=np.empty(tuple(self.output.get_shape()),dtype=np.float16)
                if getattr(self,'reuse_host_output',False):self._host_output=value
            # Opt-in caller consumes this borrowed array before the next infer.
            # All GPU reads of the previous x have finished above, so it may
            # also be the next generation's destination without an extra copy.
            self.output.copy_to(ov.Tensor(value,shared_memory=True))
            if not all_finite(value): raise ValueError("Invalid bounded FFN output")
            return value
        compiled,request = self.engines[-1]
        # Unused gate Results enforce F16 storage on GPU. Never materialize
        # those RemoteTensors; only the original public ABI may reach the host.
        outputs = ([compiled.output("x_out")] if self.role == "post" else
                   [p for p in compiled.outputs if not p.get_any_name().startswith("modulation_projected_")
                    and p.get_any_name() not in getattr(self,"remote_output_names",frozenset())])
        result = {port.get_any_name():np.array(request.get_tensor(port.get_any_name()).data,copy=True,order="C")
                  for port in outputs}
        if any(value.dtype != np.float16 or not all_finite(value) for value in result.values()):
            raise ValueError("Invalid physical modulation result")
        if getattr(self,"remote_output_names",frozenset()):
            remote_kv=getattr(self,"remote_kv",
                              {"k_device","v_device"} <= getattr(self,"remote_output_names",frozenset()))
            for name in (('q_finite','kv_finite') if remote_kv else ('q_finite',)):
                finite=result.get(name)
                if (not isinstance(finite,np.ndarray) or finite.shape!=() or finite.dtype!=np.float16
                        or not all_finite(finite) or finite.item()!=np.float16(1)):
                    raise ValueError('Invalid remote '+('Q' if name=='q_finite' else 'K/V')+' finite result')
        return result["x_out"] if self.role == "post" else result
