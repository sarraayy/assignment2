# Experiment 3 - Quantization / Precision Effect
# Comparing FP32 vs FP16

import time, csv, os, psutil, torch
from transformers import AutoTokenizer, AutoModelForCausalLM



MODEL_NAME = "distilgpt2"
PROMPT = "Artificial intelligence is transforming education"
MAX_NEW_TOKENS = 50
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULT_DIR, "experiment3_results.csv")


def measure_precision(model, tokenizer, device, prompt, max_new_tokens):
    proc = psutil.Process()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

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
        "precision": "FP16" if model.dtype == torch.float16 else "FP32",
        "elapsed_seconds": round(elapsed, 3),
        "tokens_per_sec": round(tps, 2),
        "mem_rss_before_bytes": rss_before,
        "mem_rss_after_bytes": rss_after
    }

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


    results = []


    for precision in ["fp32", "fp16"]:
        print(f"\n=== Running {precision.upper()} ===")
        dtype = torch.float16 if precision == "fp16" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        r = measure_precision(model, tokenizer, device, PROMPT, MAX_NEW_TOKENS)
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

