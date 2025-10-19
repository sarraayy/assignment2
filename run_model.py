from transformers import AutoTokenizer, AutoModelForCasualLM
import torch
import time

model_name = "distilgpt2"

print(f"Loading model: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCasualLM.from_pretrained(model)

device = "cuda" if torch.cuda.is_avaliable() else "cpu"
model.to(device)
print(f"Using device: {device}")
prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

start_time = time.time()
outputs = model.generate(**inputs, max_new_tokens=50)
end_time = time.time()

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nGenerated text:")
print(generated_text)
print(f"\nTime taken: {end_time - start_time:.2f} seconds")