import os
import json
import numpy as np

import chatts.vllm.chatts_vllm  # register ChatTS multimodal support
from vllm import LLM, SamplingParams

MODEL_PATH = os.environ.get("CHATTS_MODEL_PATH", "./ckpt")
OUT_DIR = os.environ.get("CHATTS_OUT_DIR", "./exp/chatts_puhti_demo")
os.makedirs(OUT_DIR, exist_ok=True)

# Build three synthetic time-series QA samples
x1 = np.arange(256)
ts1 = np.sin(x1 / 10) * 5.0
ts1[103:] -= 10.0

x2 = np.arange(512)
ts2 = x2 * 0.01
ts2[100] += 10.0

x3 = np.arange(256)
ts3 = np.cos(x3 / 12) * 3.0
ts3[140:170] += 5.0

samples = [
    {
        "question": "I have 2 time series. TS1: <ts><ts/> and TS2: <ts><ts/>. Please analyze trends and local changes, then tell whether both series have abrupt changes near similar times.",
        "timeseries": [ts1.tolist(), ts2.tolist()],
    },
    {
        "question": "I have 2 time series. TS1: <ts><ts/> and TS2: <ts><ts/>. Which series has a stronger short-term spike and why?",
        "timeseries": [ts2.tolist(), ts3.tolist()],
    },
    {
        "question": "I have 3 time series: cpu <ts><ts/>, memory <ts><ts/>, success <ts><ts/>. Summarize their patterns and identify possible anomaly intervals.",
        "timeseries": [ts1.tolist(), ts2.tolist(), ts3.tolist()],
    },
]

language_model = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    max_model_len=6000,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,
    limit_mm_per_prompt={"timeseries": 50},
)

sampling = SamplingParams(max_tokens=220, temperature=0.2)

results = []
for s in samples:
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{s[question]}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    req = [{"prompt": prompt, "multi_modal_data": {"timeseries": s["timeseries"]}}]
    out = language_model.generate(req, sampling_params=sampling)
    ans = out[0].outputs[0].text
    results.append({"question": s["question"], "answer": ans})

out_path = os.path.join(OUT_DIR, "generated_answers.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Saved: {out_path}")
for i, r in enumerate(results, 1):
    print(f"--- Sample {i} ---")
    print(r["answer"][:500])
