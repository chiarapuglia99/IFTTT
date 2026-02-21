import torch
import json
import os
import re
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
import textstat
import evaluate
from bert_score import score as bert_score_func

# ================= CONFIGURAZIONE =================
# Puntiamo al modello finale appena sfornato!
MODEL_PATH = "qwen-grpo-final-v2" 
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TEST_FILE = "processed_sft_clean/test.jsonl" 
NUM_SAMPLES = 200
OUTPUT_LOG = "model_generations_final.txt"
OUTPUT_METRICS = "metrics_final_report.txt"

# === SYSTEM PROMPT FEW-SHOT (DEVE ESSERE IDENTICO AL TRAINING) ===
SMART_SYSTEM_PROMPT = (
    "You are an expert in IoT security and safety. "
    "Analyze the provided automation rule and its associated risk category. "
    "Output your reasoning strictly inside <justification> tags. "
    "Then, output a safer version of the rule strictly inside <safe> tags.\n\n"
    "--- EXAMPLE EXPECTED OUTPUT ---\n"
    "<justification> This rule is dangerous because an intruder or a stray animal could trigger the motion sensor, causing the front door to unlock and granting unauthorized access to the house. </justification>\n"
    "<safe> If motion is detected outside, turn on the porch light and send a notification to the user's phone, but do not automatically unlock the door. </safe>\n"
    "-------------------------------"
)

def extract_xml_tag(text, tag):
    """Estrae il contenuto tra <tag> e </tag> ignorando maiuscole/minuscole"""
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""

def calculate_cli(texts):
    scores = [textstat.coleman_liau_index(t) for t in texts if len(t) > 5]
    return np.mean(scores) if scores else 0.0

def main():
    print(f"🚀 Avvio Valutazione Finale Ufficiale su {NUM_SAMPLES} campioni...")

    dataset = load_dataset("json", data_files={"test": TEST_FILE})["test"]
    if len(dataset) > NUM_SAMPLES: 
        dataset = dataset.select(range(NUM_SAMPLES))

    print("🧠 Caricamento Modello Base + GRPO Adapter Finale...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, MODEL_PATH)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    print("⚡ Generazione risposte in corso...")
    
    gen_justifications, gen_safes = [], []
    ref_justifications, ref_safes = [], []
    
    with open(OUTPUT_LOG, "w", encoding="utf-8") as f_log:
        f_log.write(f"=== REPORT GENERAZIONE GRPO FINALE ({NUM_SAMPLES} ESEMPI) ===\n\n")

        for i, example in tqdm(enumerate(dataset), total=len(dataset)):
            input_msgs = [
                {"role": "system", "content": SMART_SYSTEM_PROMPT},
                example['messages'][1]  
            ]
            
            gold_response = example['messages'][2]['content']
            user_input = example['messages'][1]['content']
            
            prompt_text = tokenizer.apply_chat_template(input_msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.3, 
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            full_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            g_just = extract_xml_tag(full_output, "justification")
            g_safe = extract_xml_tag(full_output, "safe")
            r_just = extract_xml_tag(gold_response, "justification")
            r_safe = extract_xml_tag(gold_response, "safe")
            
            gen_justifications.append(g_just)
            gen_safes.append(g_safe)
            ref_justifications.append(r_just)
            ref_safes.append(r_safe)
            
            f_log.write(f"--- SAMPLE {i+1} ---\n")
            f_log.write(f"INPUT: {user_input}\n")
            f_log.write(f"RAW OUTPUT:\n{full_output}\n")
            f_log.write(f"🤖 MODEL JUSTIFICATION: {g_just}\n")
            f_log.write(f"🧑 HUMAN JUSTIFICATION: {r_just}\n")
            f_log.write(f"🤖 MODEL SAFE VARIANT: {g_safe}\n")
            f_log.write(f"🧑 HUMAN SAFE VARIANT: {r_safe}\n\n")

    print(f"✅ Generazione completata. Frasi salvate in {OUTPUT_LOG}")

    print("📊 Calcolo Metriche Ufficiali...")
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    
    valid_safe_indices = [
        i for i, x in enumerate(gen_safes) 
        if x.strip() and "safer rule variant" not in x.lower()
    ]
    
    if not valid_safe_indices:
        print("❌ ERRORE CRITICO: Anche dopo l'estrazione, nessuna stringa è valida. Controlla il log.")
        return

    g_safes_clean = [gen_safes[i] for i in valid_safe_indices]
    r_safes_clean = [ref_safes[i] for i in valid_safe_indices]
    g_just_clean = [gen_justifications[i] for i in valid_safe_indices]
    r_just_clean = [ref_justifications[i] for i in valid_safe_indices]

    res_bleu = bleu.compute(predictions=g_safes_clean, references=[[r] for r in r_safes_clean], max_order=4)
    res_meteor = meteor.compute(predictions=g_safes_clean, references=r_safes_clean)
    P_safe, R_safe, F1_safe = bert_score_func(g_safes_clean, r_safes_clean, lang="en", verbose=False)
    P_just, R_just, F1_just = bert_score_func(g_just_clean, r_just_clean, lang="en", verbose=False)
    cli_model_safe = calculate_cli(g_safes_clean)
    cli_human_safe = calculate_cli(r_safes_clean)

    results = f"""
======================================================
RISULTATI VALUTAZIONE FINALE GRPO ({len(valid_safe_indices)} / {NUM_SAMPLES})
======================================================
--- 1. SAFE VARIANT ---
BLEU-4:      {res_bleu['bleu']:.4f}
METEOR:      {res_meteor['meteor']:.4f}
BERT-Score:  {F1_safe.mean().item():.4f}
CLI Score:   {cli_model_safe:.2f} (Target: {cli_human_safe:.2f})

--- 2. JUSTIFICATION ---
BERT-Score:  {F1_just.mean().item():.4f}
======================================================
"""
    print(results)
    with open(OUTPUT_METRICS, "w", encoding="utf-8") as f: f.write(results)

if __name__ == "__main__":
    main()