
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers.models.mistral.modeling_mistral import MistralMLP
from sparse_swiglu import CATS_SwiGLU, SCAP_SwiGLU
from sparse_swiglu import replace_module_with_custom, set_sparse_swiglu_threshold
from functools import partial

set_seed(42)


model_id = "mistralai/Mistral-7B-v0.1"
prompt = "I love the Avengers, "
GENLEN=32
device = "cuda"
IS_SCAP = True

cats_json = "./cfg/cats_0.5.json"
scap_json = "./cfg/scap_upgate0.4_down0.6.json"

tokenizer = AutoTokenizer.from_pretrained(model_id) # tokenizer doesnt support .to

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float32)

input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

generate_fn = partial(model.generate, 
                   min_new_tokens=GENLEN, 
                   max_new_tokens=GENLEN,
                   do_sample=False,
                   num_beams=1,
                   early_stopping=False,
                   temperature=None,
                   top_p=None)
# Dense
output_ids = generate_fn(input_ids)
generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
print(f"\nDense: {generated_texts[0]}\n\n")

if IS_SCAP is False:
    # CATS
    model = replace_module_with_custom(model, MistralMLP, CATS_SwiGLU, quiet=True)
    set_sparse_swiglu_threshold(model, cats_json)
else:
    # SCAP
    model = replace_module_with_custom(model, MistralMLP, SCAP_SwiGLU, quiet=True)
    set_sparse_swiglu_threshold(model, scap_json)

output_ids = generate_fn(input_ids)
generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
label = 'SCAP' if IS_SCAP is True else 'CATS'
print(f"\n{label}: {generated_texts[0]}\n\n")

print("end.")