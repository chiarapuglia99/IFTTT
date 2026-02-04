import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_from_disk
from bert_score import score as bert_score
from tqdm import tqdm
import numpy as np 
import evaluate  # Nuova libreria per le metriche standard
import nltk      # Necessario per METEOR
import math      # Per calcoli matematici del CLI

# Scarichiamo le risorse NLTK necessarie per METEOR
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# --- CONFIGURAZIONE ---
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_PATH = "./output_llama_qlora/final_adapter" 
DATASET_PATH = "dataset_final_processed_nobalance"

# --- CARICAMENTO NUOVE METRICHE ---
bleu_metric = evaluate.load("bleu")
meteor_metric = evaluate.load("meteor")

# --- FUNZIONE CALCOLO CLI (Readability) ---
def calculate_cli(text):
    """
    Calcola il Coleman-Liau Index per una data stringa di testo.
    Formula: CLI = 0.0588 * L - 0.296 * S - 15.8
    """
    if not text or not text.strip():
        return 0.0
    
    text = text.strip()
    letters = sum(c.isalnum() for c in text)
    words = len(text.split())
    if words == 0: return 0.0
        
    sentences = text.count('.') + text.count('!') + text.count('?')
    if sentences == 0: sentences = 1
        
    L = (letters / words) * 100
    S = (sentences / words) * 100
    
    cli_score = 0.0588 * L - 0.296 * S - 15.8
    return cli_score

# --- FUNZIONE DI SEGMENTAZIONE (INVARIATA) ---
def segment_rule_smart(text):
    text_clean = text.lower().strip()
    trigger = ""
    action = ""

    # 1. STRUTTURA DIRETTA (Causa -> Effetto)
    if " then " in text_clean:
        parts = text_clean.split(" then ", 1)
        trigger = parts[0].replace("if ", "").replace("when ", "").strip()
        action = parts[1].strip()
        return trigger, action

    # 2. STRUTTURA INVERSA (Effetto -> Causa)
    separators = [" if ", " when ", " unless ", " provided that ", " only while "]
    
    for sep in separators:
        if sep in text_clean:
            parts = text_clean.split(sep, 1)
            action = parts[0].strip()
            trigger_prefix = sep.strip() + " " if sep.strip() != "if" else ""
            trigger = trigger_prefix + parts[1].strip()
            return trigger, action

    # 3. FALLBACK
    trigger = text_clean.replace("if ", "").strip()
    action = "" 

    return trigger, action

# 1. CARICAMENTO MODELLO
print("Caricamento Tokenizer e Modello Base...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, 
    quantization_config=bnb_config,
    device_map="auto"
)

if os.path.exists(ADAPTER_PATH):
    print(f"Caricamento Adapter LoRA da {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
else:
    print("ATTENZIONE: Adapter non trovato! Valutazione sul modello base.")

model.eval()

# 2. DATASET TEST
print("Caricamento Dataset Test...")
try:
    dataset = load_from_disk(DATASET_PATH)
    test_data = dataset["test"]
except:
    print("Dataset non trovato! Esegui prima prepare_dataset.py")
    exit()

# Liste per BERT-Score (accumulo puntuale)
f1_overall = []
f1_trigger = []
f1_action = []

# Liste per CLI (accumulo puntuale)
cli_scores = []

# Liste per BLEU e METEOR (calcolo globale alla fine)
all_predictions = []
all_references = []      # Per METEOR
all_references_bleu = [] # Per BLEU

results_log = []

print(f"Valutazione su {len(test_data)} regole...")

for i in tqdm(range(len(test_data))):
    full_text_ids = test_data[i]["input_ids"]
    full_text = tokenizer.decode(full_text_ids, skip_special_tokens=False)
    
    split_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    if split_marker not in full_text:
        continue
        
    parts = full_text.split(split_marker)
    prompt_input = parts[0] + split_marker 
    ground_truth = parts[1].replace("<|eot_id|>", "").strip() 
    
    inputs = tokenizer(prompt_input, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128, 
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_full = tokenizer.decode(outputs[0], skip_special_tokens=False)
    prediction = generated_full.split(split_marker)[-1].replace("<|eot_id|>", "").strip()

    # --- RACCOLTA DATI PER METRICHE GLOBALI ---
    all_predictions.append(prediction)
    all_references.append(ground_truth)         
    all_references_bleu.append([ground_truth]) 
    
    # --- CALCOLO CLI SU QUESTA REGOLA ---
    current_cli = calculate_cli(prediction)
    cli_scores.append(current_cli)

    # --- SEGMENTAZIONE ---
    pred_trig, pred_act = segment_rule_smart(prediction)
    gt_trig, gt_act = segment_rule_smart(ground_truth)
    
    # --- CALCOLO BERT-SCORE (Tua logica originale) ---
    try:
        _, _, score_ov = bert_score([prediction], [ground_truth], lang="en", verbose=False)
        f1_overall.append(score_ov.item())
    except:
        f1_overall.append(0.0)

    try:
        _, _, score_tr = bert_score([pred_trig], [gt_trig], lang="en", verbose=False)
        f1_trigger.append(score_tr.item())
    except:
        f1_trigger.append(0.0)
    
    score_ac_val = 0.0
    if gt_act:
        if pred_act:
            _, _, score_ac = bert_score([pred_act], [gt_act], lang="en", verbose=False)
            score_ac_val = score_ac.item()
        else:
            score_ac_val = 0.0
        f1_action.append(score_ac_val)
    else:
        pass # Se GT non ha azione, non aggiungiamo nulla e non penalizziamo la media

    # --- LOGGING (Formato ESATTO richiesto) ---
    results_log.append(
        f"IDX: {i}\n"
        f"TARGET (GT): {ground_truth}\n"
        f"PREDICTION:  {prediction}\n"
        f"--- SEGMENTAZIONE ---\n"
        f"   GT Split -> [Trig]: '{gt_trig}' | [Act]: '{gt_act}'\n"
        f"   PR Split -> [Trig]: '{pred_trig}' | [Act]: '{pred_act}'\n"
        f"--- SCORES ---\n"
        f"OVERALL: {f1_overall[-1]:.4f} | TRIGGER: {f1_trigger[-1]:.4f} | ACTION: {score_ac_val:.4f} | CLI: {current_cli:.2f}\n"
        f"{'='*60}\n"
    )

# --- CALCOLO MEDIE E METRICHE GLOBALI ---
print("Calcolo metriche finali...")

avg_ov = np.mean(f1_overall) if f1_overall else 0.0
avg_tr = np.mean(f1_trigger) if f1_trigger else 0.0
avg_ac = np.mean(f1_action) if f1_action else 0.0
avg_cli = np.mean(cli_scores) if cli_scores else 0.0

# Calcolo BLEU
bleu_results = bleu_metric.compute(predictions=all_predictions, references=all_references_bleu)
final_bleu = bleu_results['bleu']

# Calcolo METEOR
meteor_results = meteor_metric.compute(predictions=all_predictions, references=all_references)
final_meteor = meteor_results['meteor']

print(f"\n✅ RISULTATI FINALI:")
print(f"BERT-Score OVERALL Mean: {avg_ov:.4f}")
print(f"BERT-Score TRIGGER Mean: {avg_tr:.4f}")
print(f"BERT-Score ACTION Mean:  {avg_ac:.4f}")
# ECCO LA RIGA CHE VOLEVI:
print(f"(Action valutata su {len(f1_action)} campioni validi su {len(test_data)})")
print(f"BLEU-4 Score:            {final_bleu:.4f}")
print(f"METEOR Score:            {final_meteor:.4f}")
print(f"CLI (Readability) Mean:  {avg_cli:.4f}")

# SALVATAGGIO FILE (Formato richiesto + sommario metriche)
with open("eval_report_final.txt", "w", encoding="utf-8") as f:
    f.write(f"=== METRICS SUMMARY ===\n")
    f.write(f"BERT-Score Overall: {avg_ov:.4f}\n")
    f.write(f"BERT-Score Trigger: {avg_tr:.4f}\n")
    f.write(f"BERT-Score Action:  {avg_ac:.4f}\n")
    f.write(f"BLEU-4 Score:       {final_bleu:.4f}\n")
    f.write(f"METEOR Score:       {final_meteor:.4f}\n")
    f.write(f"CLI Readability:    {avg_cli:.4f}\n")
    f.write(f"\n=== DETAILS ===\n")
    f.write("".join(results_log))

print("Report salvato in eval_report_final.txt")