import os
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

MODEL_PATH = os.environ.get("CHATTS_MODEL_PATH", "./ckpt")
OUT_DIR = os.environ.get("CHATTS_OUT_DIR", "./exp/chatts_hf_demo")
os.makedirs(OUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, tokenizer=tokenizer)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
model.eval()

x1 = np.arange(256)
ts1 = (np.sin(x1 / 10) * 5.0).astype(np.float32)
ts1[103:] -= 10.0
x2 = np.arange(512)
ts2 = (x2 * 0.01).astype(np.float32)
ts2[100] += 10.0
x3 = np.arange(256)
ts3 = (np.cos(x3 / 12) * 3.0).astype(np.float32)
ts3[140:170] += 5.0

samples = [
    {
        "q": "I have 2 time series. TS1: <ts><ts/> and TS2: <ts><ts/>. Analyze trends and local changes, and tell if abrupt changes happen near similar times.",
        "ts": [ts1, ts2],
    },
    {
        "q": "I have 3 time series: cpu <ts><ts/>, memory <ts><ts/>, success <ts><ts/>. Summarize patterns and likely anomaly intervals.",
        "ts": [ts1, ts2, ts3],
    },
]

results = []
for s in samples:
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{s['q']}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = processor(text=[prompt], timeseries=s["ts"], padding=True, return_tensors="pt")
    inputs = {k: v.to(0) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    gen = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    results.append({"question": s["q"], "answer": gen})

out_path = os.path.join(OUT_DIR, "generated_answers_hf.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Saved: {out_path}")
for i, r in enumerate(results, 1):
    print(f"--- Sample {i} ---")
    print(r["answer"][:500])
