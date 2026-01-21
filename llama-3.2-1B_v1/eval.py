import torch
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_from_disk
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1. Configurazione Percorsi e Modello
base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
adapter_path = "./model_final_llama" 
dataset_path = "../dataset_llama"

print("Caricamento modello e tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

# Caricamento Modello Base
model = AutoModelForCausalLM.from_pretrained(
    base_model_id, 
    torch_dtype=torch.float16, 
    device_map={"": 0}
)

# Caricamento Pesi Addestrati (Adapter LoRA)
if os.path.exists(adapter_path):
    print(f"Applicazione pesi addestrati da {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
else:
    print("Cartella adapter non trovata. Uso il modello base.")

model.eval()

# 2. Caricamento Dataset di Test
print(f"Caricamento dataset da {dataset_path}...")
dataset = load_from_disk(dataset_path)
test_data = dataset["test"]
# Selezioniamo 100 campioni per le statistiche, i primi 30 verranno scritti nel debug
#test_subset = test_data.select(range(min(100, len(test_data))))

# Stiamo utilizzando tutto il dataset di test
test_subset = dataset["test"]

y_true = []
y_pred = []
examples_output = []

print(f"Inizio inferenza su {len(test_subset)} campioni...")

for i in range(len(test_subset)):
    full_prompt = test_subset[i]["prompt"]
    true_label = str(test_subset[i]["target_category"]).strip().lower()
    
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Pulizia della risposta (rimuoviamo il prompt)
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_len = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
    response = full_text[prompt_len:].strip()
    
    # Estrazione Categoria tramite Regex
    match = re.search(r"<category>(.*?)</category>", response, re.DOTALL | re.IGNORECASE)
    if match:
        pred_label = match.group(1).strip().lower()
    else:
        # Fallback manuale per le statistiche
        if "personal" in response.lower(): pred_label = "personal harm"
        elif "physical" in response.lower(): pred_label = "physical harm"
        elif "cyber" in response.lower(): pred_label = "cybersecurity harm"
        else: pred_label = "parsing_error"
            
    y_true.append(true_label)
    y_pred.append(pred_label)
    
    # Raccolta dati per i primi 30 esempi
    if i < 30:
        examples_output.append(
            f"CAMPIONE {i+1}\n"
            f"TARGET REALE: {true_label}\n"
            f"PREDETTO: {pred_label}\n"
            f"RISPOSTA COMPLETA:\n{response}\n"
            f"{'-'*50}\n"
        )

    if (i+1) % 10 == 0:
        print(f"Avanzamento: {i+1}/{len(test_subset)}...")

# 3. Calcolo Accuracy e Report
accuracy = accuracy_score(y_true, y_pred)
unique_labels = sorted(list(set(y_true) | set(y_pred)))

# 4. Salvataggio Esempi Debug
with open("test_examples_30.txt", "w", encoding="utf-8") as f:
    f.write("=== ANALISI PRIMI 30 ESEMPI GENERATI ===\n\n")
    f.write("".join(examples_output))

# 5. Generazione Matrice di Confusione
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=unique_labels, yticklabels=unique_labels, cmap='Blues')
plt.xlabel('Predetto')
plt.ylabel('Reale')
plt.title(f'Matrice di Confusione - Llama-GRPO (Accuracy: {accuracy:.2%})')
plt.tight_layout()
plt.savefig("confusion_matrix.png")

# 6. Salvataggio Report Statistico
with open("final_report.txt", "w", encoding="utf-8") as f:
    f.write("RELAZIONE FINALE VALUTAZIONE\n")
    f.write("="*30 + "\n")
    f.write(f"ACCURACY GLOBALE: {accuracy:.2%}\n\n")
    f.write("REPORT DETTAGLIATO:\n")
    f.write(classification_report(y_true, y_pred, zero_division=0))

print(f"\n✅ Valutazione completata!")
print(f"📊 Accuracy: {accuracy:.2%}")
print(f"📄 File generati: 'final_report.txt', 'test_examples_30.txt' e 'confusion_matrix.png'")