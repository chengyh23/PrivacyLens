from transformers import AutoTokenizer, AutoModelForCausalLM

model="google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="bfloat16")
