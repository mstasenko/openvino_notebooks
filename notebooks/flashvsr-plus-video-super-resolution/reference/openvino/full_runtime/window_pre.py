"""Bound pre normalization/QKV/RoPE by whole windows, not reduced context."""
import numpy as np
from physical_modulation import Modulation

LANES=("q_windows","k_windows","v_windows")
GATES=("gate_msa","shift_mlp","scale_mlp","gate_mlp")
KV_DEVICE_SHAPE=(32,128,12,128)


def window_indices(geometry,start,count):
    """Source flat-token indices in the exact WindowPartition3D order."""
    f,h,w=geometry
    windows=f*h*w//128
    if type(start)!=int or type(count)!=int or start<0 or count<1 or start+count>windows:
        raise ValueError("Window range outside complete token geometry")
    ids=np.arange(start,start+count,dtype=np.int64)[:,None,None,None]
    spatial=(h//8)*(w//8)
    frame=(ids//spatial)*2+np.arange(2,dtype=np.int64)[None,:,None,None]
    y=((ids%spatial)//(w//8))*8+np.arange(8,dtype=np.int64)[None,None,:,None]
    x=(ids%(w//8))*8+np.arange(8,dtype=np.int64)[None,None,None,:]
    return ((frame*h+y)*w+x).reshape(-1)


def bounded_model(source,rows=3840):
    """Authenticate and clone source partitions, then schedule independent rows."""
    import openvino as ov
    from openvino import opset13 as ops
    if type(rows)!=int or rows<128 or rows>8192 or rows%128:
        raise ValueError("Pre blocks require 128-token windows and at most8192rows")
    model=source.clone()
    names={port.get_any_name()for port in model.outputs}
    allowed={*LANES,*GATES,"residual","q_pooled","k_pooled"}
    if not {*LANES,*GATES}<=names or not names<=allowed or ("q_pooled"in names)!=("k_pooled"in names):
        raise ValueError("Source pre output ABI changed")
    geometry=None;replacements=[]
    def literal(node,index,expected):
        value=node.input_value(index).get_node()
        if value.get_type_name()!="Constant" or value.output(0).get_element_type()!=ov.Type.i64 or list(value.get_vector())!=expected:
            raise ValueError("Source window shape/axes changed")
    for name in LANES:
        final=model.output(name).get_node().input_value(0).get_node()
        if final.get_type_name()!="Reshape":raise ValueError("Source window output is not Reshape")
        transpose=final.input_value(0).get_node()
        if transpose.get_type_name()!="Transpose":raise ValueError("Source window transpose changed")
        literal(transpose,1,[0,1,3,5,2,4,6,7])
        split=transpose.input_value(0).get_node()
        if split.get_type_name()!="Reshape":raise ValueError("Source window split changed")
        volume=split.input_value(0).get_node()
        if volume.get_type_name()!="Reshape":raise ValueError("Source token-volume reshape changed")
        dims=list(volume.output(0).get_shape())
        if len(dims)!=5 or dims[0]!=1 or dims[4]!=1536:
            raise ValueError("Source token-volume ABI changed")
        _,f,h,w,_=dims
        if (f not in (2,4,6) or h<8 or w<8 or h%8 or w%8 or max(h,w)>240 or h*w>136*240):
            raise ValueError("Pre geometry exceeds reviewed padded-UHD scope")
        current=(f,h,w)
        if geometry is not None and geometry!=current:raise ValueError("Q/K/V geometries differ")
        geometry=current
        literal(volume,1,dims)
        literal(split,1,[1,f//2,2,h//8,8,w//8,8,1536])
        literal(final,1,[f*h*w//128,128,1536])
        flat=volume.input_value(0)
        if flat.get_element_type()!=ov.Type.f16 or tuple(flat.get_shape())!=(1,f*h*w,1536):
            raise ValueError("Original flat Q/K/V F16 ABI changed")
        replacements.append((final,flat))
    tokens=int(np.prod(geometry));windows=tokens//128
    if tuple(model.input("x").get_shape())!=(1,tokens,1536):raise ValueError("Pre input geometry differs")
    residual="residual"in names
    if residual:
        result=model.output("residual").get_node()
        convert=result.input_value(0).get_node()
        if (convert.get_type_name()!="Convert" or convert.input_value(0)!=model.input("x")
                or convert.output(0).get_element_type()!=ov.Type.f16):
            raise ValueError("Pre residual is not original half input identity")
        model.remove_result(result)
    # Feed X and BOTH RoPE arrays in window order. Normalization/projection/
    # rotation are pointwise in token rows, so the full spatial transpose is
    # unnecessary; each output slab already has its final window order.
    views=[]
    for final,flat in replacements:
        view=ops.reshape(flat,ops.constant([windows,128,1536],ov.Type.i64),False)
        view.output(0).get_tensor().set_names(final.output(0).get_names())
        final.output(0).replace(view.output(0));views.append(view)
    view_shapes={view.input_value(1).get_node().get_instance_id()for view in views}
    for node in model.get_ordered_ops():
        if node.get_type_name()!="Constant" or node.output(0).get_element_type()!=ov.Type.i64:continue
        values=np.asarray(node.get_data())
        if node.get_instance_id()in view_shapes:
            resized=[rows//128,128,1536]
        elif values.ndim==1 and len(values)>=2 and values[0]==1 and values[1]==tokens:
            uses=list(node.output(0).get_target_inputs())
            if any((use.get_node().get_type_name(),use.get_index())not in (("Reshape",1),("StridedSlice",2))for use in uses):
                raise ValueError("Token literal has an unexpected consumer")
            resized=values.copy();resized[1]=rows
        else:continue
        node.output(0).replace(ops.constant(resized,ov.Type.i64).output(0))
    for name,shape,dtype in (("x",[1,tokens,1536],ov.Type.f16),
                             ("rope_cos",[1,tokens,1,64],ov.Type.f32),("rope_sin",[1,tokens,1,64],ov.Type.f32)):
        port=model.input(name)
        if list(port.get_shape())!=shape or port.get_element_type()!=dtype:raise ValueError("Pre data input ABI changed")
        shape[1]=rows
        port.get_node().set_partial_shape(ov.PartialShape(shape))
    model.validate_nodes_and_infer_types()
    return model,geometry,residual


class WindowPre:
    """Reuse physical pre requests and staging; keep all original output windows."""
    def __init__(self,core,device,model,config,rows=3840,remote_q=False,remote_kv=False,**precision):
        if type(remote_q)!=bool:raise ValueError("remote_q must be bool")
        if type(remote_kv)!=bool:raise ValueError("remote_kv must be bool")
        if remote_kv and not remote_q:raise ValueError("Remote K/V requires remote Q")
        graph,self.geometry,self.residual=bounded_model(model,rows)
        self.rows=rows;self.windows=int(np.prod(self.geometry))//128
        self.remote_q=remote_q;self.remote_kv=remote_kv
        self.remote_query_count=(self.windows+15)//16 if remote_q else None
        if remote_q:
            if rows!=4096 or self.windows not in (60,180,510,1530):
                raise ValueError("Remote Q requires reviewed 4096-row HD/UHD windows")
            if self.remote_query_count%2:
                raise ValueError("Remote Q requires an even C16 slot count")
            if not {"q_pooled","k_pooled"} <= {port.get_any_name() for port in graph.outputs}:
                raise ValueError("Remote Q requires source pooled Q/K outputs")
            from device_query_graph import expose_device_queries
            if remote_kv:
                expose_device_queries(graph,remote_kv=True)
            else:
                expose_device_queries(graph)
        worker_precision=dict(precision)
        if remote_kv:
            worker_precision["remote_kv"]=True
        self.worker=Modulation(core,device,graph,config,"pre",**worker_precision)
        self.context=self.worker.context;self.engines=self.worker.engines
        self.weight_names=self.worker.weight_names;self.feed_names=self.worker.feed_names
        # The tail pair is lazy in the context, but reserve its exact resident
        # accounting up front so reports do not understate the remote-K/V path.
        self.kv_scratch_bytes=(2*int(np.prod(KV_DEVICE_SHAPE))*2) if remote_kv else 0
        self.bytes=self.worker.bytes+self.kv_scratch_bytes
        self.staging={"x":np.empty((1,rows,1536),np.float16),
                      **{name:np.empty((1,rows,1,64),np.float32)for name in ("rope_cos","rope_sin")}}
        self.host_staging_bytes=sum(value.nbytes for value in self.staging.values())
        self.pooled={port.get_any_name()for port in graph.outputs if port.get_any_name().endswith("_pooled")}
        self._kv_scratch=None

    @staticmethod
    def _kv_cache_spec(geometry,windows):
        """Return the full cache rows and fresh suffix offset for this geometry."""
        if (not isinstance(geometry,(tuple,list)) or len(geometry)!=3 or
                type(windows)!=int or windows<=0):
            raise ValueError("Invalid K/V cache geometry")
        frames=geometry[0]
        if type(frames) is not int:
            raise ValueError("K/V cache requires integer frame geometry")
        if frames==6:
            total=windows
        elif frames==2:
            total=4*windows
        else:
            raise ValueError("K/V cache requires two- or six-frame geometry")
        return total,total-windows

    @staticmethod
    def _remote_value_shape(value):
        try:
            shape=tuple(value.get_shape())
            byte_size=int(value.get_byte_size())
        except Exception as error:
            raise ValueError("Remote K/V cache must expose shape and byte size") from error
        # The pinned RemoteTensor binding exposes shape/bytes, but not always
        # get_element_type; canonical F16 is authenticated by this exact byte
        # contract, with an optional type check on fakes/newer bindings.
        element_type=getattr(value,"get_element_type",None)
        if element_type is not None:
            import openvino as ov
            if element_type()!=ov.Type.f16:
                raise ValueError("Remote K/V cache must be F16")
        return shape,byte_size

    def _validate_kv_remote_slots(self,slots,q_remote_slots=None):
        """Validate full canonical roots; per-window ROI views are made below."""
        import openvino as ov
        if not isinstance(slots,tuple) or len(slots)!=2:
            raise ValueError("Remote K/V requires the complete K_cache/V_cache pair")
        if id(slots[0])==id(slots[1]):
            raise ValueError("Remote K/V cache roots must be distinct")
        remote_type=getattr(ov,"RemoteTensor",None)
        if remote_type is None or any(not isinstance(value,remote_type) for value in slots):
            raise ValueError("Remote K/V cache roots must be RemoteTensor objects")
        total,offset=self._kv_cache_spec(self.geometry,self.windows)
        expected=(total,128,12,128)
        expected_bytes=int(np.prod(expected))*2
        devices=[];contexts=[]
        for value in slots:
            shape,byte_size=self._remote_value_shape(value)
            if shape!=expected or byte_size!=expected_bytes:
                raise ValueError("Remote K/V cache root shape or byte size changed")
            get_device=getattr(value,"get_device_name",None)
            if get_device is not None:
                devices.append(get_device())
            get_params=getattr(value,"get_params",None)
            if get_params is not None:
                try:
                    params=get_params()
                    marker=(params.get("OCL_CONTEXT") if hasattr(params,"get") else None)
                    if marker is None:
                        try:
                            marker=params["OCL_CONTEXT"]
                        except Exception:
                            pass
                except Exception:
                    marker=None
                if marker is not None:
                    contexts.append(marker)
        if len(devices)==2 and devices[0]!=devices[1]:
            raise ValueError("Remote K/V cache roots must share their device context")
        if (len(contexts)==2 and contexts[0] is not contexts[1]
                and contexts[0] != contexts[1]):
            raise ValueError("Remote K/V cache roots must share their owning context")
        if q_remote_slots is not None and any(id(value) in {id(slot) for slot in q_remote_slots}
                                              for value in slots):
            raise ValueError("Remote K/V cache roots must not alias remote Q slots")
        return total,offset

    @staticmethod
    def _remote_roi(root,start,count):
        import openvino as ov
        return ov.RemoteTensor(root,ov.Coordinate([start,0,0,0]),
                               ov.Coordinate([start+count,128,12,128]))

    def _kv_scratch_pair(self):
        pair=getattr(self,"_kv_scratch",None)
        if pair is None:
            import openvino as ov
            context=getattr(self,"context",None)
            if context is None or not hasattr(context,"create_tensor"):
                raise ValueError("Remote K/V tail requires an owned GPU context")
            pair=tuple(context.create_tensor(ov.Type.f16,ov.Shape(KV_DEVICE_SHAPE),{})
                        for _ in range(2))
            if id(pair[0])==id(pair[1]):
                raise ValueError("Remote K/V tail scratch buffers must be distinct")
            self._kv_scratch=pair
        for value in pair:
            shape,byte_size=self._remote_value_shape(value)
            if shape!=KV_DEVICE_SHAPE or byte_size!=int(np.prod(KV_DEVICE_SHAPE))*2:
                raise ValueError("Remote K/V tail scratch ABI changed")
        return pair

    @staticmethod
    def _require_finite_one(value,name):
        if (not isinstance(value,np.ndarray) or value.shape!=() or value.dtype!=np.float16
                or not np.isfinite(value).all() or value.item()!=np.float16(1)):
            raise ValueError(f"Invalid per-window-pre {name} result")

    def _copy_kv_to_host(self,roots,offset,start,count,result):
        """Read one canonical ROI directly into the existing source-layout result."""
        import openvino as ov
        for name,root in zip(("k_windows","v_windows"),roots):
            destination=result[name][start:start+count].reshape((count,128,12,128))
            self._remote_roi(root,offset+start,count).copy_to(
                ov.Tensor(destination,shared_memory=True))

    def infer(self,feeds,weights,q_remote_slots=None,kv_remote_slots=None,keep_host_kv=True):
        if type(keep_host_kv)!=bool:
            raise ValueError("keep_host_kv must be bool")
        remote_q=getattr(self,"remote_q",False)
        remote_kv=getattr(self,"remote_kv",False)
        if not keep_host_kv and (not remote_q or not remote_kv or
                                 not {"q_pooled","k_pooled"} <= getattr(self,"pooled",set())):
            raise ValueError("keep_host_kv=False requires remote Q/KV and pooled outputs")
        if remote_q:
            if not isinstance(q_remote_slots,tuple) or len(q_remote_slots)!=self.remote_query_count:
                raise ValueError("Remote Q requires the complete C16 slot tuple")
            if len({id(slot) for slot in q_remote_slots})!=len(q_remote_slots):
                raise ValueError("Remote Q slots must be distinct")
        elif q_remote_slots is not None:
            raise ValueError("Remote Q slots are not valid for ordinary window pre")
        if remote_kv:
            if not remote_q:
                raise ValueError("Remote K/V requires remote Q")
            _kv_total,kv_offset=self._validate_kv_remote_slots(kv_remote_slots,q_remote_slots)
        elif kv_remote_slots is not None:
            raise ValueError("Remote K/V slots are not valid for ordinary window pre")
        if set(feeds)!=self.feed_names:raise ValueError("Complete window-pre feed fields changed")
        tokens=self.windows*128
        for name,value in feeds.items():
            shape,dtype=((1,tokens,1536),np.float16)if name=="x"else ((1,6,1536),np.float16)if name=="t_mod"else ((1,tokens,1,64),np.float32)
            if not isinstance(value,np.ndarray)or value.shape!=shape or value.dtype!=dtype:
                raise ValueError("Complete window-pre shape/dtype changed")
        result=({name:np.empty((self.windows,128,1536),np.float16)
                 for name in (("k_windows","v_windows") if remote_q else LANES)}
                if keep_host_kv else {})
        result.update({name:np.empty((self.windows,1536),np.float16)for name in self.pooled})
        for start in range(0,self.windows,self.rows//128):
            count=min(self.rows//128,self.windows-start)
            indices=window_indices(self.geometry,start,count);valid=count*128
            for name,slab in self.staging.items():
                # Indices are generated inside the validated source geometry.
                # Non-raising mode avoids NumPy's extra buffered output copy.
                np.take(feeds[name],indices,axis=1,out=slab[:,:valid],mode="clip")
                slab[:,valid:]=1 if name=="rope_cos"else 0
            remote_outputs=None
            if remote_q:
                slots=q_remote_slots[start//16:start//16+2]
                remote_outputs={"q_device_0":slots[0],"q_device_1":slots[1]}
            kv_targets=None
            if remote_kv:
                if count==self.rows//128:
                    kv_targets=tuple(self._remote_roi(root,kv_offset+start,count)
                                     for root in kv_remote_slots)
                else:
                    kv_targets=self._kv_scratch_pair()
                remote_outputs.update({"k_device":kv_targets[0],"v_device":kv_targets[1]})
            if remote_q:
                current=self.worker.infer({**self.staging,"t_mod":feeds["t_mod"]},weights,
                                          remote_outputs=remote_outputs)
            else:
                current=self.worker.infer({**self.staging,"t_mod":feeds["t_mod"]},weights)
            if remote_q:
                finite=current.get("q_finite")
                if (not isinstance(finite,np.ndarray) or finite.shape!=() or
                        finite.dtype!=np.float16 or not np.isfinite(finite).all() or
                        finite.item()!=np.float16(1)):
                    raise ValueError("Invalid per-window-pre q_finite result")
            if remote_kv:
                self._require_finite_one(current.get("kv_finite"),"kv_finite")
                if count!=self.rows//128:
                    for root,scratch in zip(kv_remote_slots,kv_targets):
                        self._remote_roi(root,kv_offset+start,count).copy_from(
                            self._remote_roi(scratch,0,count))
                if keep_host_kv:
                    self._copy_kv_to_host(kv_remote_slots,kv_offset,start,count,result)
            else:
                for name in (("k_windows","v_windows") if remote_q else LANES):
                    result[name][start:start+count]=current[name][:count]
            for name in self.pooled:result[name][start:start+count]=current[name][:count]
            for name in GATES:
                if start==0:result[name]=current[name]
                elif result[name].tobytes()!=current[name].tobytes():raise ValueError("Window scheduling changed modulation gates")
        if self.residual:result["residual"]=np.array(feeds["x"],copy=True,order="C")
        return result
