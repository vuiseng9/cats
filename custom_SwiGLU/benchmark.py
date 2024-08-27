
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers.models.mistral.modeling_mistral import MistralMLP
from sparse_swiglu import CATS_SwiGLU, SCAP_SwiGLU
from sparse_swiglu import replace_module_with_custom, set_sparse_swiglu_threshold
from functools import partial
import json
import time
import types
import inspect
from tqdm import tqdm

set_seed(42)

def summarize(l):
    return dict(
        mean = sum(l)/len(l),
        median = sorted(l)[len(l)//2],
        min = min(l),
        max = max(l))

model_id = "mistralai/Mistral-7B-v0.1"
prompt = "I love the Avengers, "
device = "cuda"
IS_SCAP = False
IS_SCAP = True
IS_DENSE = False

GENLEN=64
N_WARMUPS = 5
N_LOOP = 10

cats_json = "./cfg/cats_0.5.json"
scap_json = "./cfg/scap_upgate0.4_down0.6.json"

tokenizer = AutoTokenizer.from_pretrained(model_id) # tokenizer doesnt support .to
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float32)

input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

label = 'DENSE'
if IS_DENSE is False:
    if IS_SCAP is False:
        # CATS
        label = 'CATS'
        model = replace_module_with_custom(model, MistralMLP, CATS_SwiGLU, quiet=True)
        set_sparse_swiglu_threshold(model, cats_json)
    else:
        # SCAP
        label = 'SCAP'
        model = replace_module_with_custom(model, MistralMLP, SCAP_SwiGLU, quiet=True)
        set_sparse_swiglu_threshold(model, scap_json)


token_elapses = []

model_forward = model.forward
def bench_forward(self, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.time()
    
    ret = model_forward.__func__(self, *args, **kwargs)

    torch.cuda.synchronize()
    t1 = time.time()

    token_elapses.append(t1 - t0)
    return ret

bench_forward.__signature__ = inspect.signature(model.forward)
model.forward = types.MethodType(bench_forward, model)

bench_fn = partial(model.generate, 
                   min_new_tokens=GENLEN, 
                   max_new_tokens=GENLEN,
                   do_sample=False,
                   num_beams=1,
                   early_stopping=False,
                   temperature=None,
                   top_p=None)

for _ in tqdm(range(N_WARMUPS), desc="Warming up "):
    bench_fn(input_ids)

token_elapses = [] # emptying
for _ in tqdm(range(N_LOOP), desc="Benchmarking "):
    output_ids = bench_fn(input_ids)


generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
print(f"\n{label}: {generated_texts[0]}\n\n")

assert len(token_elapses) // GENLEN == N_LOOP, "token_elapses must have GENLEN * N_LOOP elements"

TTFT_list = []
ITL_list = []
for i, el in enumerate(token_elapses):
    if i % GENLEN == 0:
        TTFT_list.append(el)
    else:
        ITL_list.append(el)

itl_summary = summarize(ITL_list)
print(f"{label}: Gen. Len: {GENLEN}, Warmup: {N_WARMUPS}, Bench Loop: {N_LOOP}\n"\
      f"TTFT:{json.dumps(summarize(TTFT_list), indent=4)}\n"\
      f"ITL:{json.dumps(summarize(ITL_list), indent=4)}")

print("end")