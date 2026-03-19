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

# Try to load bleurt, fallback if not available
try:
    from bleurt import score as bleurt_score
    HAS_BLEURT = True
except ImportError:
    HAS_BLEURT = False
    print("BLEURT non installato. Installa con: pip install bleurt")
except Exception as e:
    HAS_BLEURT = False
    print(f"Errore caricamento BLEURT: {e}")

# ================= CONFIGURAZIONE =================
# Puntiamo al modello finale appena sfornato!
MODEL_PATH = "qwen-grpo-final-v2" 
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TEST_FILE = "processed_sft_clean/test.jsonl" 
NUM_SAMPLES = 200  # Campioni per valutazione finale

# Log outputs - Base directory for all results
RESULTS_DIR = "evaluation_results"
OUTPUT_LOG = f"{RESULTS_DIR}/model_generations_final.txt"
OUTPUT_METRICS = f"{RESULTS_DIR}/summary.txt"

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

def setup_metrics_dirs():
    """Crea la struttura di cartelle per i risultati"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    metrics_dirs = {
        'bleu_safe': f"{RESULTS_DIR}/bleu/safe_variant",
        'bleu_just': f"{RESULTS_DIR}/bleu/justification",
        'meteor_safe': f"{RESULTS_DIR}/meteor/safe_variant",
        'meteor_just': f"{RESULTS_DIR}/meteor/justification",
        'bert_score_safe': f"{RESULTS_DIR}/bert_score/safe_variant",
        'bert_score_just': f"{RESULTS_DIR}/bert_score/justification",
        'cli_score_safe': f"{RESULTS_DIR}/cli_score/safe_variant",
        'cli_score_just': f"{RESULTS_DIR}/cli_score/justification",
        'bleurt_safe': f"{RESULTS_DIR}/bleurt/safe_variant",
        'bleurt_just': f"{RESULTS_DIR}/bleurt/justification",
    }
    
    for key, dir_path in metrics_dirs.items():
        os.makedirs(dir_path, exist_ok=True)
    
    return metrics_dirs

def save_metric_result(metric_dir, metric_name, data_dict):
    """Salva i risultati di una metrica in formato JSON e TXT"""
    # Salva JSON
    json_path = f"{metric_dir}/results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)
    
    # Salva TXT
    txt_path = f"{metric_dir}/results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"=== METRICA: {metric_name} ===\n\n")
        for key, value in data_dict.items():
            if isinstance(value, (list, dict)):
                f.write(f"{key}:\n{json.dumps(value, indent=2, ensure_ascii=False)}\n\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"✅ Risultati {metric_name} salvati in {metric_dir}/")


def calculate_cli(texts):
    scores = [textstat.coleman_liau_index(t) for t in texts if len(t) > 5]
    return np.mean(scores) if scores else 0.0

def main():
    print(f"🚀 Avvio Valutazione Finale Ufficiale su {NUM_SAMPLES} campioni...")
    
    # Setup directories
    metrics_dirs = setup_metrics_dirs()
    print(f"📁 Struttura di cartelle creata in {RESULTS_DIR}/")

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
        print(" ERRORE CRITICO: Anche dopo l'estrazione, nessuna stringa è valida. Controlla il log.")
        return

    g_safes_clean = [gen_safes[i] for i in valid_safe_indices]
    r_safes_clean = [ref_safes[i] for i in valid_safe_indices]
    g_just_clean = [gen_justifications[i] for i in valid_safe_indices]
    r_just_clean = [ref_justifications[i] for i in valid_safe_indices]

    # === Calcolo BLEU e METEOR ===
    res_bleu = bleu.compute(predictions=g_safes_clean, references=[[r] for r in r_safes_clean], max_order=4)
    res_meteor = meteor.compute(predictions=g_safes_clean, references=r_safes_clean)
    
    # Calcolo BLEU e METEOR per Justifications
    res_bleu_just = bleu.compute(predictions=g_just_clean, references=[[r] for r in r_just_clean], max_order=4)
    res_meteor_just = meteor.compute(predictions=g_just_clean, references=r_just_clean)
    
    # Salva BLEU Safe Variant
    bleu_data_safe = {
        "metric": "BLEU (Safe Variant)",
        "bleu_score": float(res_bleu['bleu']),
        "precisions": res_bleu.get('precisions', []),
        "samples_count": len(g_safes_clean)
    }
    save_metric_result(metrics_dirs['bleu_safe'], "BLEU-4 (Safe)", bleu_data_safe)
    
    # Salva BLEU Justification
    bleu_data_just = {
        "metric": "BLEU (Justification)",
        "bleu_score": float(res_bleu_just['bleu']),
        "precisions": res_bleu_just.get('precisions', []),
        "samples_count": len(g_just_clean)
    }
    save_metric_result(metrics_dirs['bleu_just'], "BLEU-4 (Justification)", bleu_data_just)
    
    # Salva METEOR Safe Variant
    meteor_data_safe = {
        "metric": "METEOR (Safe Variant)",
        "meteor_score": float(res_meteor['meteor']),
        "samples_count": len(g_safes_clean)
    }
    save_metric_result(metrics_dirs['meteor_safe'], "METEOR (Safe)", meteor_data_safe)
    
    # Salva METEOR Justification
    meteor_data_just = {
        "metric": "METEOR (Justification)",
        "meteor_score": float(res_meteor_just['meteor']),
        "samples_count": len(g_just_clean)
    }
    save_metric_result(metrics_dirs['meteor_just'], "METEOR (Justification)", meteor_data_just)
    
    # === Calcolo BERT-Score ===
    P_safe, R_safe, F1_safe = bert_score_func(g_safes_clean, r_safes_clean, lang="en", verbose=False)
    P_just, R_just, F1_just = bert_score_func(g_just_clean, r_just_clean, lang="en", verbose=False)
    
    # Salva BERT-Score Safe Variant
    bert_safe_data = {
        "metric": "BERT-Score (Safe Variant)",
        "precision_mean": float(P_safe.mean().item()),
        "recall_mean": float(R_safe.mean().item()),
        "f1_mean": float(F1_safe.mean().item()),
        "precision_list": P_safe.tolist(),
        "recall_list": R_safe.tolist(),
        "f1_list": F1_safe.tolist(),
        "samples_count": len(g_safes_clean)
    }
    save_metric_result(metrics_dirs['bert_score_safe'], "BERT-Score (Safe)", bert_safe_data)
    
    # Salva BERT-Score Justification
    bert_just_data = {
        "metric": "BERT-Score (Justification)",
        "precision_mean": float(P_just.mean().item()),
        "recall_mean": float(R_just.mean().item()),
        "f1_mean": float(F1_just.mean().item()),
        "precision_list": P_just.tolist(),
        "recall_list": R_just.tolist(),
        "f1_list": F1_just.tolist(),
        "samples_count": len(g_just_clean)
    }
    save_metric_result(metrics_dirs['bert_score_just'], "BERT-Score (Justification)", bert_just_data)
    
    # === Calcolo CLI-Score (Coleman-Liau Index) ===
    cli_model_safe = calculate_cli(g_safes_clean)
    cli_human_safe = calculate_cli(r_safes_clean)
    cli_model_just = calculate_cli(g_just_clean)
    cli_human_just = calculate_cli(r_just_clean)
    
    # Salva CLI-Score Safe Variant
    cli_safe_data = {
        "metric": "CLI-Score (Safe Variant)",
        "model_cli": float(cli_model_safe),
        "reference_cli": float(cli_human_safe),
        "description": "Coleman-Liau Index: Basato su lettere, numeri e spazi (Target ≈ 12-14)",
        "samples_count": len(g_safes_clean)
    }
    save_metric_result(metrics_dirs['cli_score_safe'], "CLI-Score (Safe)", cli_safe_data)
    
    # Salva CLI-Score Justification
    cli_just_data = {
        "metric": "CLI-Score (Justification)",
        "model_cli": float(cli_model_just),
        "reference_cli": float(cli_human_just),
        "description": "Coleman-Liau Index: Basato su lettere, numeri e spazi (Target ≈ 12-14)",
        "samples_count": len(g_just_clean)
    }
    save_metric_result(metrics_dirs['cli_score_just'], "CLI-Score (Justification)", cli_just_data)
    
    # === Calcolo BLEURT-Score ===
    bleurt_scores_safe = None
    bleurt_scores_just = None
    if HAS_BLEURT:
        try:
            bleurt_scorer = bleurt_score.BleurtScorer()
            bleurt_scores_safe = bleurt_scorer.score(
                references=r_safes_clean,
                candidates=g_safes_clean,
                batch_size=16
            )
            bleurt_scores_just = bleurt_scorer.score(
                references=r_just_clean,
                candidates=g_just_clean,
                batch_size=16
            )
            bleurt_mean_safe = np.mean(bleurt_scores_safe)
            bleurt_mean_just = np.mean(bleurt_scores_just)
            
            # Salva BLEURT-Score Safe Variant
            bleurt_safe_data = {
                "metric": "BLEURT-Score (Safe Variant)",
                "mean_score": float(bleurt_mean_safe),
                "scores": bleurt_scores_safe,
                "min_score": float(np.min(bleurt_scores_safe)),
                "max_score": float(np.max(bleurt_scores_safe)),
                "std_score": float(np.std(bleurt_scores_safe)),
                "samples_count": len(g_safes_clean)
            }
            save_metric_result(metrics_dirs['bleurt_safe'], "BLEURT-Score (Safe)", bleurt_safe_data)
            
            # Salva BLEURT-Score Justification
            bleurt_just_data = {
                "metric": "BLEURT-Score (Justification)",
                "mean_score": float(bleurt_mean_just),
                "scores": bleurt_scores_just,
                "min_score": float(np.min(bleurt_scores_just)),
                "max_score": float(np.max(bleurt_scores_just)),
                "std_score": float(np.std(bleurt_scores_just)),
                "samples_count": len(g_just_clean)
            }
            save_metric_result(metrics_dirs['bleurt_just'], "BLEURT-Score (Justification)", bleurt_just_data)
        except Exception as e:
            print(f"⚠️ Errore calcolo BLEURT: {e}")
            bleurt_scores_safe = None
            bleurt_scores_just = None
    else:
        print("⚠️ BLEURT non disponibile, metriche saltate")

    results = f"""
======================================================
RISULTATI VALUTAZIONE FINALE GRPO ({len(valid_safe_indices)} / {NUM_SAMPLES})
======================================================

✅ Tutti i risultati dettagliati sono salvati nella cartella: {RESULTS_DIR}/

📊 STRUTTURA RISULTATI:
───────────────────────
{RESULTS_DIR}/
├── bleu/
│   ├── safe_variant/
│   │   ├── results.json
│   │   └── results.txt
│   └── justification/
│       ├── results.json
│       └── results.txt
├── meteor/
│   ├── safe_variant/
│   │   ├── results.json
│   │   └── results.txt
│   └── justification/
│       ├── results.json
│       └── results.txt
├── bert_score/
│   ├── safe_variant/
│   │   ├── results.json
│   │   └── results.txt
│   └── justification/
│       ├── results.json
│       └── results.txt
├── cli_score/
│   ├── safe_variant/
│   │   ├── results.json
│   │   └── results.txt
│   └── justification/
│       ├── results.json
│       └── results.txt
├── bleurt/
│   ├── safe_variant/
│   │   ├── results.json
│   │   └── results.txt
│   └── justification/
│       ├── results.json
│       └── results.txt
├── model_generations_final.txt
└── summary.txt

📈 RIEPILOGO METRICHE:
──────────────────────
--- 1. SAFE VARIANT ---
BLEU-4:          {res_bleu['bleu']:.4f}
METEOR:          {res_meteor['meteor']:.4f}
BERT-Score F1:   {F1_safe.mean().item():.4f}
CLI Score:       {cli_model_safe:.2f} (Reference: {cli_human_safe:.2f})
""" + (f"BLEURT-Score:    {bleurt_mean_safe:.4f}\n" if HAS_BLEURT and bleurt_scores_safe else "") + f"""
--- 2. JUSTIFICATION ---
BLEU-4:          {res_bleu_just['bleu']:.4f}
METEOR:          {res_meteor_just['meteor']:.4f}
BERT-Score F1:   {F1_just.mean().item():.4f}
CLI Score:       {cli_model_just:.2f} (Reference: {cli_human_just:.2f})
""" + (f"BLEURT-Score:    {bleurt_mean_just:.4f}\n" if HAS_BLEURT and bleurt_scores_just else "") + f"""
======================================================
Note:
- Ogni metrica ha una cartella dedicata con risultati in JSON e TXT
- I file JSON contengono i dati completi per analisi ulteriore
- CLI Score (Coleman-Liau Index): Target ≈ 12-14 (text universitario-tecnico)
- BERT-Score: Correlazione semantica con riferimento
- BLEURT-Score: Valutazione qualitativa basata su modello (se disponibile)
=====================================================
"""
    print(results)
    with open(OUTPUT_METRICS, "w", encoding="utf-8") as f: 
        f.write(results)
    
    print(f"\n✅ Riepilogo completo salvato in {OUTPUT_METRICS}")

if __name__ == "__main__":
    main()