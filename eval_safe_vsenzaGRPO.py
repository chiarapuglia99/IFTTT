import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_from_disk
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# --- CONFIGURAZIONE ---
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_PATH = "./output_llama_standard_vsenzaGRPO/final_adapter" 
DATASET_PATH = "dataset_final_processed"
CSV_PATH = "dataset.csv" 

# 1. CARICAMENTO MODELLO
print("Caricamento Tokenizer e Modello Base...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" # Importante per generazione batch (se usata) o singola

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

if os.path.exists(ADAPTER_PATH):
    print(f"Caricamento Adapter LoRA da {ADAPTER_PATH}...")
    try:
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        model = model.merge_and_unload()
    except Exception as e:
        print(f"Errore caricamento adapter: {e}")
else:
    print("ATTENZIONE: Adapter non trovato! Valutazione sul modello base.")

model.eval()

# 2. PREPARAZIONE DATI TEST
print("Caricamento Dataset Test...")
dataset = load_from_disk(DATASET_PATH)
test_data = dataset["test"]

# Caricamento CSV per riferimenti (opzionale, per label belle)
risk_map = {1: "Personal Harm", 2: "Physical Harm", 3: "Cybersecurity Harm"}
try:
    df_ref = pd.read_csv(CSV_PATH)
except:
    df_ref = None

# Modelli metriche
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
sem_model = SentenceTransformer('all-MiniLM-L6-v2')

results = []
rouge_scores = []
sem_scores = []

print(f"Inizio Valutazione su {len(test_data)} campioni...")

# Loop di valutazione
for i in tqdm(range(len(test_data))):
    # Decodifichiamo l'intero esempio (che contiene System + User + Assistant)
    full_text_ids = test_data[i]["input_ids"]
    full_text = tokenizer.decode(full_text_ids, skip_special_tokens=False)
    
    # SPEZZIAMO IL PROMPT DALLA RISPOSTA ATTESA
    # Il nostro prompt finisce con: "<|start_header_id|>assistant<|end_header_id|>\n\n"
    split_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    if split_marker not in full_text:
        continue # Skip se malformato
        
    parts = full_text.split(split_marker)
    prompt_input = parts[0] + split_marker 
    ground_truth = parts[1].replace("<|eot_id|>", "").strip() 
    
    # GENERAZIONE
    inputs = tokenizer(prompt_input, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128, 
            do_sample=False,       
            temperature=0.0,       
            pad_token_id=tokenizer.eos_token_id
        )
        
    # Decodifica output
    generated_full = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Estraiamo solo la parte nuova
    prediction = generated_full.split(split_marker)[-1].replace("<|eot_id|>", "").strip()

    # METRICHE
    r_score = scorer.score(ground_truth, prediction)['rougeL'].fmeasure
    
    emb1 = sem_model.encode(ground_truth, convert_to_tensor=True)
    emb2 = sem_model.encode(prediction, convert_to_tensor=True)
    s_score = util.pytorch_cos_sim(emb1, emb2).item()
    
    rouge_scores.append(r_score)
    sem_scores.append(s_score)
    
    # Recupero info extra dal CSV (euristico) per il report
    cat_str = "Unknown"
    trigger_str = "N/A"
    action_str = "N/A"
    
    if df_ref is not None:
        # Cerchiamo nel CSV la riga che ha ESATTAMENTE questa safe rule target
        match = df_ref[df_ref['safe'].str.strip() == ground_truth]
        if not match.empty:
            cat_str = risk_map.get(match.iloc[0]['label'], "Unknown")
            # Aggiunta estrazione trigger e action dal dataset
            trigger_str = match.iloc[0]['triggerTitle']
            action_str = match.iloc[0]['actionTitle']
            
    # Salvataggio log con trigger e action inclusi
    results.append(
        f"IDX: {i}\n"
        f"TRIGGER: {trigger_str}\n"
        f"ACTION:  {action_str}\n"
        f"CATEGORY: {cat_str}\n"
        f"TARGET: {ground_truth}\n"
        f"PRED:   {prediction}\n"
        f"SCORES: ROUGE={r_score:.4f} | SEM={s_score:.4f}\n"
        f"{'-'*60}\n"
    )

# 3. REPORT FINALE
avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
avg_sem = sum(sem_scores) / len(sem_scores) if sem_scores else 0

print(f"\n✅ Valutazione Terminata.")
print(f"Media ROUGE-L: {avg_rouge:.4f}")
print(f"Media Semantic: {avg_sem:.4f}")

with open("eval_report_standard.txt", "w", encoding="utf-8") as f:
    f.write(f"=== REPORT VALUTAZIONE (Trainer Standard) ===\n")
    f.write(f"Modello: {ADAPTER_PATH}\n")
    f.write(f"Media ROUGE-L: {avg_rouge:.4f}\n")
    f.write(f"Media SEMANTIC: {avg_sem:.4f}\n\n")
    f.write("".join(results))

# 4. GRAFICO
plt.figure(figsize=(10, 6))
plt.hist(rouge_scores, bins=20, alpha=0.5, label='ROUGE-L', color='blue')
plt.hist(sem_scores, bins=20, alpha=0.5, label='Semantic Similarity', color='green')
plt.axvline(avg_rouge, color='blue', linestyle='dashed', linewidth=2)
plt.axvline(avg_sem, color='green', linestyle='dashed', linewidth=2)
plt.xlabel('Score')
plt.ylabel('Count')
plt.title('Distribuzione Scores: ROUGE vs Semantic')
plt.legend()
plt.savefig("eval_chart_standard.png")
print("Grafico salvato in eval_chart_standard.png")