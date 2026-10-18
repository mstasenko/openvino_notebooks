"""Diagnostic CPU metadata reset only; validate all 32 original sites first."""
EXPECTED=tuple('blocks.'+str(i)+'.attn.rope.rope' for i in range(32))
MODULE='rotary_embedding_torch.rotary_embedding_torch'
def require(value,message):
    if not value:raise ValueError(message)
def validate_rows(rows):
    require(type(rows) is list and len(rows)==32,'Expected 32 reset records')
    for name,row in zip(EXPECTED,rows):
        require(set(row)=={'name','before','after'} and row['name']==name and
                type(row['before']) is int and 0<=row['before']<=8192 and row['after']==0 and type(row['after']) is int,
                'Malformed reset metadata')
    return rows
def reset(model):
    sites=[(name,module) for name,module in model.named_modules() if type(module).__name__=='RotaryEmbedding']
    require(tuple(name for name,module in sites)==EXPECTED,'Rotary module count/role differs')
    rows=[]
    for name,module in sites:
        require(type(module).__module__==MODULE,'Rotary implementation type differs')
        require(module.cache_if_possible is True and module.learned_freq is False and
                module.freqs_for=='lang' and module.cache_max_seq_len==8192 and module.use_xpos is False,
                'Unsupported rotary cache configuration')
        require(tuple(module.cached_freqs.shape)==(8192,42) and tuple(module.freqs.shape)==(21,) and
                str(module.cached_freqs.dtype)==str(module.freqs.dtype)=='torch.bfloat16' and
                str(module.cached_freqs.device)==str(module.freqs.device)=='cpu',
                'Unsupported rotary cache/frequency schema or placement')
        length=module.cached_freqs_seq_len
        require(type(length) is int and 0<=length<=8192,'Malformed cached frequency length')
        rows.append({'name':name,'before':length,'after':0})
    # Do not zero/replace tensors or change cache policy; next original forward
    # misses its length guard and recomputes the same original frequency math.
    for name,module in sites:module.cached_freqs_seq_len=0
    return validate_rows(rows)
