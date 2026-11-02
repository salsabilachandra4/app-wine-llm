"""Klasifikasi kultivar anggur (UCI Wine) dengan LLM lewat Groq API.

File yang SAMA dipakai saat evaluasi (notebook Colab) dan saat deploy (Streamlit),
supaya perilaku yang di-deploy persis sama dengan yang dievaluasi.
Client Groq dibuat di luar file ini (Colab: dari Colab Secrets, Streamlit: dari st.secrets).
"""
import json
import re
import time

from groq import RateLimitError

FITUR = ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
         "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
         "color_intensity", "hue", "od280_od315", "proline"]

# catatan pemakaian (diisi classify, dibaca saat evaluasi)
USAGE = {"request": 0, "detik": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
         "rate_limit": 0, "terpotong": 0}


def reset_usage():
    for k in USAGE:
        USAGE[k] = 0.0 if k == "detik" else 0


def load_config(path="config.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_data(row):
    """Satu baris data anggur -> teks satu baris untuk LLM."""
    return ", ".join(f"{k}={float(row[k]):g}" for k in FITUR)


def build_messages(cfg, data):
    msgs = [{"role": "system", "content": cfg["system_prompt"]}]
    for ex in cfg.get("few_shot", []):                     # contoh berlabel, boleh kosong
        msgs.append({"role": "user", "content": format_data(ex["data"])})
        msgs.append({"role": "assistant", "content": ex["label"]})
    msgs.append({"role": "user", "content": format_data(data)})
    return msgs


def parse_label(teks, labels):
    """'Kultivar 2.' / 'kultivar_2' -> 'Kultivar_2'. Di luar daftar label -> 'invalid'."""
    if not teks:
        return "invalid"
    bersih = re.sub(r"[^a-z0-9]", "", teks.lower())
    peta = {re.sub(r"[^a-z0-9]", "", l.lower()): l for l in labels}
    if bersih in peta:                                     # cocok persis
        return peta[bersih]
    muncul = {peta[k] for k in peta if k in bersih}        # tepat satu label disebut
    return muncul.pop() if len(muncul) == 1 else "invalid"


def classify(client, cfg, data, max_retry=4):
    kwargs = dict(model=cfg["model"], messages=build_messages(cfg, data),
                  temperature=cfg["temperature"],
                  max_completion_tokens=cfg["max_tokens"])
    if cfg.get("reasoning_effort"):
        kwargs["extra_body"] = {"reasoning_effort": cfg["reasoning_effort"]}
    for percobaan in range(max_retry):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(**kwargs)
            USAGE["detik"] += time.time() - t0
            USAGE["request"] += 1
            if resp.usage:
                USAGE["prompt_tokens"] += resp.usage.prompt_tokens
                USAGE["completion_tokens"] += resp.usage.completion_tokens
            if resp.choices[0].finish_reason == "length":
                USAGE["terpotong"] += 1
            return parse_label(resp.choices[0].message.content, cfg["labels"])
        except RateLimitError:
            USAGE["rate_limit"] += 1
            time.sleep(15 * (percobaan + 1))               # kena kuota: tunggu, coba lagi
    return "invalid"
