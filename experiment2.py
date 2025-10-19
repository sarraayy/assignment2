# Experiment 2 - Batching Effect
# Compare performance for batch sizes 1, 2, and 4

import time, csv, os, psutil, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "distilgpt2"
PROMPTS = [
    "Once upon a time",
    "Artificial intelligence is transforming education",
    "Music has the power to",
    "The future of robotics depends on human collaboration" 

]

BATCH_SIZES = [1, 2, 4]
MAX_NEW_TOKENS = 50
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULT_DIR, "experiment2_results.csv")

def measure_batch(model, tokenizer, device, prompts, max_new_tokens):
    proc = psutil.Process()

    # Tokenize as a batch
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
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
    elapsed = t1 - t0
    total_generated = out.shape[-1] - inputs["input_ids"].shape[-1]
    throughput = total_generated / elapsed if elapsed > 0 else 0.0

    return {
        "batch_size": len(prompts),
        "elapsed_seconds": round(elapsed, 3),
        "tokens_per_sec": round(throughput, 2),
        "mem_rss_before_bytes": rss_before,
        "mem_rss_after_bytes": rss_after
    }

def main():
    print("Loading model:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    results = []

    for batch_size in BATCH_SIZES:
        current_prompts = PROMPTS[:batch_size]
        print(f"\n=== Batch size = {batch_size} ===")
        r = measure_batch(model, tokenizer, device, current_prompts, MAX_NEW_TOKENS)
        results.append(r)
        print(f"Done: {r['elapsed_seconds']}s | {r['tokens_per_sec']} toks/s")


    # Saving results
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nResults written to:", CSV_PATH)
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
