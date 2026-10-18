"""Compact the shared key union of adjacent queries, preserving every mask bit."""
from dataclasses import replace
import numpy as np
from query_axis import build_query_axis_graph,memory_bound,MAX_GRAPH_BYTES


def capacities(maximum,*,fine=False):
    if type(maximum)!=int or maximum<=0:raise ValueError('Positive union capacity required')
    if type(fine)!=bool:raise ValueError('Explicit boolean fine-bucket mode required')
    # Research UHD layouts: add intermediate capacities without exceeding the
    # ORIGINAL combined six-GiB inventory (including every live request).
    if fine and maximum in (512,640):
        return (128,256,384,512)if maximum==512 else(224,352,640)
    small=320 if maximum<=512 else 384
    return ((small,)if small<maximum else())+(maximum,)


def plan(mask,chunk,slots):
    if (not isinstance(mask,np.ndarray) or mask.dtype!=np.bool_ or mask.ndim!=3
            or any(n<=0 for n in mask.shape) or type(chunk)!=int or chunk<=0
            or not slots or tuple(sorted(set(slots)))!=tuple(slots)
            or any(type(v)!=int or not 0<v<=mask.shape[-1]for v in slots)):
        raise ValueError('Boolean mask, positive query batch and ordered union capacities required')
    result=[]
    for start in range(0,mask.shape[1],chunk):
        count=min(chunk,mask.shape[1]-start)
        needed=max(1,int(mask[:,start:start+count].any(1).sum(1).max()))
        capacity=next((value for value in slots if value>=needed),None)
        if capacity is None:raise ValueError('Union capacity would discard allowed keys')
        result.append((start,count,capacity))
    return result


def bucket_inventory(spec,slots):
    if not slots or tuple(sorted(set(slots)))!=tuple(slots):raise ValueError('Ordered unique capacities required')
    rows=[inventory(spec,n)for n in slots]
    # Only the canonical device K/V tensors are shared between requests.
    total=sum(r['bounded_total_bytes']for r in rows)-(len(rows)-1)*rows[0]['source_cache_bytes']
    if total>MAX_GRAPH_BYTES:raise MemoryError('Combined union bucket inventory exceeds six GiB')
    return total


def pack(mask,capacity):
    if (not isinstance(mask,np.ndarray) or mask.dtype!=np.bool_ or mask.ndim!=3
            or any(n<=0 for n in mask.shape) or type(capacity)!=int or not 0<capacity<=mask.shape[-1]):
        raise ValueError('Expected boolean [head,query,key] and bounded integer capacity')
    h,c,_=mask.shape
    ids=np.zeros((h,capacity),np.int32)
    compact=np.zeros((h,c,capacity),np.bool_)
    for head in range(h):
        selected=np.flatnonzero(mask[head].any(0))
        if len(selected)>capacity:raise ValueError('Union capacity would discard allowed keys')
        ids[head,:len(selected)]=selected
        compact[head,:,:len(selected)]=mask[head][:,selected]
    return ids,compact


def inventory(spec,capacity):
    if type(capacity)!=int or not 0<capacity<=spec.key_windows:
        raise ValueError('Invalid union capacity')
    bounded=replace(spec,key_windows=capacity)
    result=memory_bound(bounded,batch_queries=True)
    # The base inventory already includes gathered K/V. Full source caches
    # remain resident as well; views/reshape do not copy them.
    result['source_cache_bytes']=2*int(np.prod(spec.kv_shape))*2
    result['selection_bytes']=spec.heads*capacity*4
    result['bounded_total_bytes']+=result['source_cache_bytes']+result['selection_bytes']
    result['bounded_total_gib']=result['bounded_total_bytes']/2**30
    return result


def build(spec,capacity,actual_logit_bound):
    import openvino as ov
    from openvino import opset13 as ops
    bound=inventory(spec,capacity)
    if bound['bounded_total_bytes']>MAX_GRAPH_BYTES:
        raise MemoryError('Union attention exceeds six-GiB structural bound')
    graph=build_query_axis_graph(replace(spec,key_windows=capacity),
                                 actual_logit_bound=actual_logit_bound,batch_queries=True)
    ids=ops.parameter([spec.heads,capacity],ov.Type.i32,name='selectedKeyBlock')
    ids.output(0).get_tensor().set_names({'selectedKeyBlock'})
    offsets=(np.arange(spec.heads,dtype=np.int32)*spec.key_windows)[:,None]
    if spec.heads*spec.key_windows>=2**31:raise ValueError('Head-local gather index overflows int32')
    indices=ops.add(ids,ops.constant(offsets))
    for name in ('K_cache','V_cache'):
        original=graph.input(name).get_node()
        source=ops.parameter(spec.kv_shape,ov.Type.f16,name=name)
        source.output(0).get_tensor().set_names({name})
        flat=ops.reshape(source,ops.constant([spec.heads*spec.key_windows,spec.block_tokens*spec.depth],ov.Type.i64),False)
        gathered=ops.gather(flat,indices,ops.constant(np.int64(0)),name='Union_'+name)
        value=ops.reshape(gathered,ops.constant([spec.heads,capacity,spec.block_tokens,spec.depth],ov.Type.i64),False)
        original.output(0).replace(value.output(0))
        value.output(0).get_tensor().set_names(set())
        graph.remove_parameter(original);graph.add_parameters([source])
    graph.add_parameters([ids]);graph.validate_nodes_and_infer_types()
    return graph
