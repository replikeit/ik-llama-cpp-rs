#!/usr/bin/env python
"""Assemble a combined NextN/MTP GGUF for ik_llama.cpp from two Thireus halves.

base   = general merged model (32-layer metadata + tokenizer + token_embd/output_norm + blk.0..31)
inject = mtp merged model's blk.32.* tensors (transformer block 32 + nextn head)
patch  = qwen35.block_count 32->33, add qwen35.nextn_predict_layers=1

Result: one self-contained 33-layer NextN model runnable with `--spec-type mtp` (no -md).
"""
import sys
import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType

GEN = sys.argv[1]
MTP = sys.argv[2]
OUT = sys.argv[3]
ARCH = "qwen35"

def decode_field(field):
    if field and field.types:
        t = field.types[0]
        if t == GGUFValueType.ARRAY:
            st = field.types[-1]
            if st == GGUFValueType.STRING:
                return [str(bytes(field.parts[i]), encoding='utf-8') for i in field.data]
            return [pv for i in field.data for pv in field.parts[i].tolist()]
        if t == GGUFValueType.STRING:
            return str(bytes(field.parts[-1]), encoding='utf-8')
        return field.parts[-1][0]
    return None

gen = GGUFReader(GEN)
mtp = GGUFReader(MTP)

writer = GGUFWriter(OUT, ARCH)

OVERRIDE = {
    f"{ARCH}.block_count": (33, GGUFValueType.UINT32),
    f"{ARCH}.nextn_predict_layers": (1, GGUFValueType.UINT32),
}
added = set()

# 1) copy general KV (skip virtual + architecture), applying overrides
for field in gen.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
        continue
    if field.name in OVERRIDE:
        val, vt = OVERRIDE[field.name]
        writer.add_key_value(field.name, val, vt)
        added.add(field.name)
        print(f"  override {field.name} = {val}")
        continue
    val = decode_field(field)
    if val is not None:
        writer.add_key_value(field.name, val, field.types[0])
# add any overrides not present in general (e.g. nextn_predict_layers)
for k, (val, vt) in OVERRIDE.items():
    if k not in added:
        writer.add_key_value(k, val, vt)
        print(f"  add {k} = {val}")

# 2) tensor infos: all general tensors, then mtp's blk.32.* tensors
plan = []
for t in gen.tensors:
    plan.append(("gen", t))
n_gen = len(plan)
inj = 0
for t in mtp.tensors:
    if t.name.startswith("blk.32."):
        plan.append(("mtp", t))
        inj += 1
print(f"  general tensors: {n_gen}, injected blk.32.* from mtp: {inj}, total: {len(plan)}")
# sanity: nextn tensors present?
nextn = [t.name for _, t in plan if "nextn" in t.name]
print(f"  nextn tensors: {nextn}")

for _, t in plan:
    writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
for _, t in plan:
    writer.write_tensor_data(t.data)
writer.close()
print(f"WROTE {OUT}")
