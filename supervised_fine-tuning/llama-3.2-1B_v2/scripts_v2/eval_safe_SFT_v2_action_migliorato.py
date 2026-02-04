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

# --- CONFIGURAZIONE ---
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_PATH = "./output_llama_qlora/final_adapter" 
DATASET_PATH = "dataset_final_processed_nobalance"

# --- FUNZIONE DI SEGMENTAZIONE MIGLIORATA (SUPER SMART) ---
def segment_rule_smart(text):
    """
    Versione potenziata: riconosce più connettivi logici per ridurre i 'falsi zeri'.
    Supporta: IF, THEN, WHEN, UNLESS, PROVIDED.
    """
    text_clean = text.lower().strip()
    trigger = ""
    action = ""

    # 1. STRUTTURA DIRETTA (Causa -> Effetto)
    # Cerca "THEN" (Struttura classica TAP)
    if " then " in text_clean:
        parts = text_clean.split(" then ", 1)
        trigger = parts[0].replace("if ", "").replace("when ", "").strip()
        action = parts[1].strip()
        return trigger, action

    # 2. STRUTTURA INVERSA (Effetto -> Causa)
    # Lista di parole chiave che introducono la condizione
    separators = [" if ", " when ", " unless ", " provided that ", " only while "]
    
    for sep in separators:
        if sep in text_clean:
            # Dividiamo alla prima occorrenza del separatore
            # Esempio: "Turn on lights [when] you arrive"
            parts = text_clean.split(sep, 1)
            
            # Parte 0: Azione ("Turn on lights")
            action = parts[0].strip()
            
            # Parte 1: Trigger ("you arrive") + reinseriamo il separatore se serve contesto
            # (es. "unless" cambia il senso, quindi lo teniamo nel trigger)
            trigger_prefix = sep.strip() + " " if sep.strip() != "if" else ""
            trigger = trigger_prefix + parts[1].strip()
            
            return trigger, action

    # 3. FALLBACK (Nessun separatore trovato)
    # Se non troviamo nulla, consideriamo tutto trigger (o potremmo provare euristica imperativa)
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

f1_overall = []
f1_trigger = []
f1_action = []
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

    # --- SEGMENTAZIONE SMART ---
    pred_trig, pred_act = segment_rule_smart(prediction)
    gt_trig, gt_act = segment_rule_smart(ground_truth)
    
    # --- CALCOLO BERT-SCORE ---
    # 1. OVERALL
    try:
        _, _, score_ov = bert_score([prediction], [ground_truth], lang="en", verbose=False)
        f1_overall.append(score_ov.item())
    except:
        f1_overall.append(0.0)

    # 2. TRIGGER
    try:
        _, _, score_tr = bert_score([pred_trig], [gt_trig], lang="en", verbose=False)
        f1_trigger.append(score_tr.item())
    except:
        f1_trigger.append(0.0)
    
    # 3. ACTION (Calcoliamo solo se il Ground Truth ha un'azione valida)
    score_ac_val = 0.0
    if gt_act:
        if pred_act:
            # Confronta azione predetta con azione reale
            _, _, score_ac = bert_score([pred_act], [gt_act], lang="en", verbose=False)
            score_ac_val = score_ac.item()
        else:
            # Ground truth aveva un'azione, ma noi non l'abbiamo predetta -> 0
            score_ac_val = 0.0
        f1_action.append(score_ac_val)
    else:
        # Se il Ground Truth NON ha azione (segmentazione fallita), non penalizziamo la media con uno 0
        # Saltiamo questo campione per la media Action
        pass

    # LOGGING
    results_log.append(
        f"IDX: {i}\n"
        f"TARGET (GT): {ground_truth}\n"
        f"PREDICTION:  {prediction}\n"
        f"--- SEGMENTAZIONE ---\n"
        f"   GT Split -> [Trig]: '{gt_trig}' | [Act]: '{gt_act}'\n"
        f"   PR Split -> [Trig]: '{pred_trig}' | [Act]: '{pred_act}'\n"
        f"--- SCORES ---\n"
        f"OVERALL: {f1_overall[-1]:.4f} | TRIGGER: {f1_trigger[-1]:.4f} | ACTION: {score_ac_val:.4f}\n"
        f"{'='*60}\n"
    )

# Report Finale
avg_ov = np.mean(f1_overall) if f1_overall else 0.0
avg_tr = np.mean(f1_trigger) if f1_trigger else 0.0
avg_ac = np.mean(f1_action) if f1_action else 0.0 # Ora esclude i casi dove non c'era azione nel GT

print(f"\n✅ RISULTATI FINALI:")
print(f"BERT-Score OVERALL Mean: {avg_ov:.4f}")
print(f"BERT-Score TRIGGER Mean: {avg_tr:.4f}")
print(f"BERT-Score ACTION Mean:  {avg_ac:.4f}")
print(f"(Action valutata su {len(f1_action)} campioni validi su {len(test_data)})")

# CORREZIONE ERRORE SCRITTURA FILE
with open("eval_report_final.txt", "w", encoding="utf-8") as f:
    f.write(f"=== METRICS ===\nOverall: {avg_ov}\nTrigger: {avg_tr}\nAction: {avg_ac}\n\n")
    # Convertiamo la lista in una stringa unica prima di scrivere
    f.write("".join(results_log))

print("Report salvato in eval_report_final.txt")