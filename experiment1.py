# Experiment 1: Run generation twice: once with use_cache=True and once with use_cache=False. 
# Measure total generation time, tokens/sec, and memory usage. 
# Record differences in latency and speed.

import time
import csv
import os 
from typing import Dict, Any
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Config
MODEL_NAME = "distilgpt2"
PROMPT = "Once upon a time"
MAX_NEW_TOKENS = 50
NUM_RUNS = 3
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULT_DIR, "experiment1_results.csv")

def measure_once(model, tokenizer, device, prompt: str, max_new_tokens: int, use_cache: bool) -> Dict[str, Any]:
    proc = psutil.Process()

    # Preparing inputs
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Measurement
    rss_before = proc.memory_info().rss
    t0 = time.perf_counter()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )

    t1 = time.perf_counter()
    rss_after = proc.memory_info().rss

    generated_tokens = out.shape[-1] - inputs["input_ids"].shape[-1]
    elapsed = t1 - t0
    tps = generated_tokens / elapsed if elapsed > 0 else 0.0

    return {
        "model": MODEL_NAME,
        "use_cache": use_cache,
        "generated_tokens": int(generated_tokens),
        "elapsed_seconds": float(elapsed),
        "tokens_per_sec": float(tps),
        "mem_rss_before_bytes": int(rss_before),
        "mem_rss_after_bytes": int(rss_after),
    }


def main():
    print("Loading model and tokenizer:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    results = []
    for use_cache in [True, False]:
        print(f"\n=== use_cache = {use_cache} ===")
        for run in range(1, NUM_RUNS + 1):
            print(f"Run {run}/{NUM_RUNS} ...", end="", flush=True)
            r = measure_once(model, tokenizer, device, PROMPT, MAX_NEW_TOKENS, use_cache)
            results.append(r)
            print(f" done: {r['generated_tokens']} tokens, {r['elapsed_seconds']:.3f}s, {r['tokens_per_sec']:.1f} tok/s")



    # CSV
    fieldnames = list(results[0].keys()) if results else []
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print("\nResults written to:", CSV_PATH)
    print("Sample rows:")
    for r in results:
        print(r)

if __name__ == "__main__":
    main()



