#!/usr/bin/env python3
"""CPU-only shape/bounded-stage preparation for authenticated video geometry."""
import argparse
import gc
import json
from pathlib import Path
import resource
import sys
import time

from real_inputs import BASE, WIDTH, HEIGHT, SPATIAL_WINDOWS, load_inputs, sha
from source_precision_check import load
from frame_geometry import FrameGeometry


def parameter_modules():
    root = BASE / "openvino/parameter_layers"
    contract = load("real_parameter_contract", root / "contract.py")
    prior = sys.modules.get("contract")
    sys.modules["contract"] = contract
    try:
        builders = load("real_parameter_builders",root / "builders.py")
    finally:
        if prior is None:
            sys.modules.pop("contract",None)
        else:
            sys.modules["contract"] = prior
    full = load("real_geometry_contract",BASE / "openvino/full_model/contract.py")
    return contract, builders, full


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--attention-dtype",choices=("f32","f16"),default="f32")
    parser.add_argument("--gather-layout",choices=("nd","flat","masked"),default="nd")
    parser.add_argument("--query-chunk-windows",type=int,choices=(1,4),default=1)
    parser.add_argument("--fused-qualification",type=Path)
    parser.add_argument('--frame-size',choices=('3840x2160','2160x3840'),help='explicit visible size; original HD fixture remains the default')
    args = parser.parse_args()
    if args.query_chunk_windows != 1 and (args.attention_dtype != "f16" or args.gather_layout != "flat"):
        parser.error("C4 is qualified only for native F16 flat cache gathers")
    if args.gather_layout == "masked" and args.attention_dtype != "f16":
        parser.error("Full-cache masking requires native F16")
    visible=None
    if args.frame_size:
        w,h=map(int,args.frame_size.split('x'));visible=FrameGeometry(h,w)
        if args.fused_qualification or args.attention_dtype!='f16' or args.query_chunk_windows!=4:
            parser.error('Variable geometry requires F16/C4 bounded whole-post preparation')
    height,width=(HEIGHT,WIDTH)if visible is None else visible.working_hw
    spatial_windows=(height//128)*(width//128)
    args.output.mkdir(parents=True,exist_ok=False)
    if visible is None:
        fixtures,provenance = load_inputs()
        del fixtures
    else:provenance={'scope':'variable geometry shape preparation only; no model input or quality claim'}
    import openvino as ov
    if not ov.get_version().startswith("2026.3."):
        raise ValueError("Pinned OpenVINO2026.3 required")
    contract,builders,full = parameter_modules()
    chunk = load("real_attention_builder",BASE / "openvino/chunked_attention/builder.py")
    cache = load("real_source_cache_builder",BASE / "openvino/source_cross_cache/qualify.py")
    fused = None
    if args.fused_qualification is not None:
        root = args.fused_qualification.resolve()
        if root != (BASE / "openvino/ffn_slabs/fused-cross-qualification-20261013-r3").resolve() or sha(root / "results.json") != "f47d1dd10d3a256f0ddf94ed3a9ef932af02f201b5a89408858fd49510b2ea06":
            raise ValueError("Reviewed fused-cross CPU r3 required")
        fused = json.loads((root / "results.json").read_text())
        if fused["status"] != "pass":
            raise ValueError("Fused-cross CPU qualification failed")
    parameter_receipt = BASE / "openvino/parameter_layers/cpu-qualification-attempt10/results.json"
    saved = json.loads(parameter_receipt.read_text())
    if sha(parameter_receipt) != "bd9f369439c13de2d02412eddd5ac3dfbb7b3f9689353dad071183e72a12cdca" or saved["status"] != "pass":
        raise ValueError("Original reusable-parameter CPU proof changed")
    for key,name in (("parameter_contract","contract.py"),("parameter_builders","builders.py")):
        if sha(BASE / "openvino/parameter_layers" / name) != saved["reviewed_sources"][key]["sha256"]:
            raise ValueError("Reusable parameter producer changed")
    report = {"status":"initializing","scope":f'working {width}x{height} CPU shapes/bounded compilation; no inference/GPU/quality admission',
              "runtime":ov.get_version(),"input_provenance":provenance,"models":[],"attention_dtype":args.attention_dtype,
              "gather_layout":args.gather_layout,
              "query_chunk_windows":args.query_chunk_windows,
              'frame_geometry':None if visible is None else {'visible_hw':[visible.height,visible.width],'working_hw':[height,width]},
              'bounded_pipeline_required':visible is not None,
              "post_variant":"fused_cross_slabs" if fused else "whole",
              "script_sha256":sha(Path(__file__)),"parameter_receipt_sha256":sha(parameter_receipt),
              "producer_sha256":{str(path):sha(path) for path in (
                  BASE / "openvino/parameter_layers/builders.py", BASE / "openvino/chunked_attention/builder.py",
                  BASE / "openvino/source_cross_cache/qualify.py")}}
    core = ov.Core()
    config = {"INFERENCE_PRECISION_HINT":ov.Type.f16,"NUM_STREAMS":ov.properties.streams.Num(1),
              "PERFORMANCE_HINT":ov.properties.hint.PerformanceMode.LATENCY,"INFERENCE_NUM_THREADS":2,"CACHE_DIR":""}
    def persist():
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (args.output / "results.json").write_text(json.dumps(report,indent=2)+"\n")
    def qualify(role,geometry,model,compile_models=None):
        path = args.output / f"{role}_{geometry}.xml"
        ov.serialize(model,str(path),str(path.with_suffix(".bin")))
        start = time.perf_counter()
        stages=[model]if compile_models is None else compile_models
        for stage in stages:
            compiled=core.compile_model(stage,'CPU',config)
            if list(compiled.get_property('EXECUTION_DEVICES'))!=['CPU']:
                raise ValueError('Preparation unexpectedly selected another device')
            del compiled;gc.collect()
        report["models"].append({"role":role,"geometry":geometry,"xml":str(path.resolve()),
            "xml_sha256":sha(path),"bin_sha256":sha(path.with_suffix(".bin")),
            "compile_seconds":time.perf_counter()-start,"input_names":[p.get_any_name() for p in model.inputs],
            "output_names":[p.get_any_name() for p in model.outputs],
            'cpu_compile_scope':'whole'if compile_models is None else 'bounded stages'if stages else 'metadata only; original UHD attention GPU gate exists',
            'compiled_stages':len(stages)})
        gc.collect()
        persist()
    def authenticated_model(row):
        path = Path(row["xml"])
        if not path.is_absolute():
            path = BASE.parents[1] / path
        if sha(path) != row["xml_sha256"] or sha(path.with_suffix(".bin")) != row["bin_sha256"]:
            raise ValueError("Qualified post/FFN IR changed")
        return core.read_model(path)
    persist()
    try:
        for frames in (6,2):
            geometry = full.GeometrySpec(f'f{frames}_{width}x{height}',frames,height//16,width//16)
            geometry.validate()
            if geometry.spatial_windows != spatial_windows or geometry.attended_windows != (3 if frames==6 else 4)*spatial_windows:
                raise ValueError("Temporal/spatial cache contract changed")
            for role in ("pre","post"):
                if role=="post" and fused:
                    row, = [value["ir"] for value in fused["real_geometry"]["variants"] if value["geometry"]["tokens"]==geometry.tokens]
                    qualify(role,geometry.name,authenticated_model(row))
                else:
                    model=builders.build_parameterized(role,geometry)
                    stages=None
                    if visible is not None:
                        from physical_modulation import split,block_post_graphs
                        if role=='pre':
                            from window_pre import bounded_model
                            pooled=model.clone()
                            pooling=load('variable_pre_pooling',BASE/'openvino/mask_pool_probe.py')
                            pooling.add_pre_pool_results(pooled)
                            bounded,_,_=bounded_model(pooled,4096)
                            stages=split(bounded,'pre',norm_f32=True,qk_store=True)
                        else:
                            from token_batch_post import rewrite_post
                            from fused_post_cross import rewrite_post as cross_rewrite
                            bounded=model.clone()
                            rewrite_post(bounded,('self','ffn'));cross_rewrite(bounded,source_half=True,bounded_rows=3840)
                            stages=block_post_graphs(split(bounded,'post',gates=True,inline_gelu=True,linear_store=True),geometry.tokens,3840)
                    qualify(role,geometry.name,model,stages)
            spec = chunk.ChunkSpec(query_windows=geometry.query_windows,key_windows=geometry.attended_windows,
                                   query_chunk_windows=args.query_chunk_windows,
                                   selected_slot_capacity=geometry.attended_windows)
            if visible is not None:
                sys.path.insert(0,str(BASE/'openvino/chunked_attention'))
                from union_attention import build as union_build,inventory as union_inventory
                capacity=512 if frames==6 else 640
                spec=chunk.ChunkSpec(query_windows=geometry.query_windows,key_windows=geometry.attended_windows,
                                     query_chunk_windows=4,selected_slot_capacity=1)
                inventory=union_inventory(spec,capacity)
                attention=union_build(spec,capacity,chunk.MASKED_LOGIT_LIMIT)
            else:
                inventory = chunk.structural_memory_bound(spec,dtype_bytes=2 if args.attention_dtype=="f16" else 4,gather_layout=args.gather_layout)
                attention=chunk.build_chunked_graph(spec,state_dtype=args.attention_dtype,gather_layout=args.gather_layout)
            # C4 multiplies transient gathers/scores, not the retained cache.
            # This fixed HD experiment has a 3GiB structural cap; it is not
            # an allocator guarantee, UHD admission, or automatic default.
            budget = (6<<30)if visible is not None else (3 << 30) if args.query_chunk_windows == 4 else 5 * (1 << 28)
            if inventory["bounded_total_bytes"] > budget:
                raise ValueError("Finite attention structural budget exceeded")
            qualify('attention',geometry.name,attention,[]if visible is not None else None)
            report["models"][-1]["structural_memory_inventory"] = inventory
            report["models"][-1]["query_chunk_windows"] = args.query_chunk_windows
        for role in ("text","kv"):
            qualify(role,"shared",cache.build(role))
        if fused:
            proof = BASE / "openvino/ffn_slabs/cpu-qualification-20261013-r3/results.json"
            if sha(proof) != "0a3d6fa8241d1a6e87d11684f5a681ec35f7f72ee8d6b73a8dd7e6c5d82f2ea6":
                raise ValueError("Qualified FFN slab proof changed")
            row, = [row for row in json.loads(proof.read_text())["serialized_parameter_ir"] if row["stem"]=="post_ffn_slab_rows4096"]
            qualify("ffn","shared",authenticated_model(row))
            report["post_cpu_proof_sha256"] = sha(args.fused_qualification / "results.json")
        # Prefix/suffix were already serialized/CPU compiled at this geometry;
        # authenticate them instead of copying their constant weights again.
        if visible is not None:
            adapters=load('variable_geometry_edges',BASE/'openvino/full_model/builders.py')
            weights,receipt=full._EDGES.load_weights()
            if receipt['checkpoint_sha256']!=full.CHECKPOINT_SHA256:raise ValueError('Original edge weights changed')
        else:
            edges = BASE / "openvino/full_model/ir-20261013-1280x768-subset-r2"
            manifest = json.loads((edges / "manifest.json").read_text())
        for phase in ("startup","steady"):
            for role in ("prefix","suffix"):
                if visible is not None:
                    frames=6 if phase=='startup'else 2
                    geometry=full.GeometrySpec(f'{phase}_f{frames}_{width}x{height}',frames,height//16,width//16)
                    qualify(role,f'f{frames}_{width}x{height}',getattr(adapters,'build_'+role)(weights,geometry))
                    continue
                key = f"{phase}_{role}.xml"
                rec = manifest["graphs"][key]
                path = edges / key
                if sha(path) != rec["xml_sha256"] or sha(path.with_suffix(".bin")) != rec["bin_sha256"]:
                    raise ValueError("Real edge IR changed")
                report["models"].append({"role":role,"geometry":"f6_1280x768" if phase=="startup" else "f2_1280x768",
                    "xml":str(path),"xml_sha256":rec["xml_sha256"],"bin_sha256":rec["bin_sha256"],
                    "existing_geometry_cpu_compile_evidence":str(edges / "cpu-smoke.json")})
        report["memory_plan"] = {
            "retained_host_f16_kv_bytes":30*2*3*spatial_windows*128*1536*2,
            "single_layer_f16_kv_steady_bytes":2*4*spatial_windows*128*1536*2,
            "startup_ffn_single_hidden_f16_bytes":6*(height//16)*(width//16)*8960*2,
            "largest_attention_structural_bytes":max(row.get("structural_memory_inventory",{}).get("bounded_total_bytes",0) for row in report["models"]),
            "allocator_peak_proven":False,"UHD_admitted":False}
        report["status"] = "pass"
    except BaseException as error:
        report.update(status="error",error=repr(error))
        raise
    finally:
        core = None
        gc.collect()
        persist()
    print(json.dumps({"status":report["status"],"models":len(report["models"]),"peak_rss_kib":report["peak_rss_kib"]}))


if __name__ == "__main__":
    main()
