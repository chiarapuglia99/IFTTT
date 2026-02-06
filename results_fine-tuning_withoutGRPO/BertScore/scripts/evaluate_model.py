import torch
import os
import matplotlib.pyplot as plt
import ast
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_from_disk
from bert_score import score
from tqdm import tqdm

# --- CONFIGURAZIONE ---
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_PATH = "./output_llama_standard_vsenzaGRPO/final_adapter" 
DATASET_PATH = "dataset_final_processed"
LOG_FILE = "training_log.txt"
EVAL_REPORT_FILE = "eval_report_bertscore.txt"
LOSS_PLOT_FILE = "loss_plot.png"

def plot_training_loss(log_file, output_img):
    """
    Versione ROBUSTA: Legge il log, gestisce righe spezzate e pulisce i dati
    prima di generare il grafico Loss vs Epochs.
    """
    train_epochs = []
    train_losses = []
    eval_epochs = []
    eval_losses = []
    
    if not os.path.exists(log_file):
        print(f"ERRORE: File log {log_file} non trovato. Impossibile tracciare il grafico.")
        return

    print(f"Analisi del file di log: {log_file}...")

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 1. Pulizia: rimuove eventuali tag se presenti (artefatti di copia/incolla)
        content = re.sub(r'\\', '', content)

        # 2. Regex per trovare tutto ciò che sta tra graffe dopo "METRICS:"
        # re.DOTALL permette al punto (.) di includere anche i caratteri di nuova riga (\n)
        pattern = re.compile(r"METRICS:\s*(\{.*?\})", re.DOTALL)
        matches = pattern.findall(content)
        
        print(f"✅ Trovati {len(matches)} blocchi di metriche.")

        for dict_str in matches:
            try:
                # Rimuove a capo e spazi extra per rendere la stringa parsabile
                clean_str = dict_str.replace('\n', ' ').replace('\r', '')
                # Fix per eventuali 'nan' che fanno crashare ast
                if 'nan' in clean_str:
                    clean_str = clean_str.replace('nan', 'None')
                
                data = ast.literal_eval(clean_str)
                
                # --- Estrazione Dati Training ---
                if 'loss' in data and 'epoch' in data:
                    if data['loss'] is not None:
                        train_epochs.append(data['epoch'])
                        train_losses.append(data['loss'])
                
                # --- Estrazione Dati Validation ---
                if 'eval_loss' in data and 'epoch' in data:
                    if data['eval_loss'] is not None:
                        eval_epochs.append(data['epoch'])
                        eval_losses.append(data['eval_loss'])

            except Exception as e:
                # Ignora blocchi corrotti occasionali
                continue

    except Exception as e:
        print(f"Errore critico durante la lettura del file: {e}")
        return
    
    # 3. Creazione del Grafico
    if not train_epochs and not eval_epochs:
        print("⚠️ Nessun dato valido trovato per il grafico.")
        return

    plt.figure(figsize=(10, 6))
    
    if train_epochs:
        plt.plot(train_epochs, train_losses, label='Training Loss', color='blue', alpha=0.6)
    if eval_epochs:
        plt.plot(eval_epochs, eval_losses, label='Validation Loss', color='red', linewidth=2, marker='o')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(output_img)
    print(f"🎉 Grafico salvato in {output_img}")

def generate_safe_rule(model, tokenizer, description, justification, label_str):
    # Ricostruisce il prompt user come in training
    user_msg = (
        f"DESCRIPTION: {description}\n"
        f"JUSTIFICATION: {justification}\n"
        f"LABEL: {label_str}"
    )
    
    # Prompt formattato
    input_text = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are an expert AI security assistant. Your task is to generate a 'safe' automation rule based on the description, justification, and risk category provided.\n"
        "Input Format:\n"
        "DESCRIPTION: <description of the rule>\n"
        "JUSTIFICATION: <security justification>\n"
        "LABEL: <risk category>\n\n"
        "Output:\n"
        "SAFE_MODEL: <safe automation rule><|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Estrazione della parte dopo "assistant"
    if "assistant" in generated:
        parts = generated.split("assistant")
        clean_output = parts[-1].strip()
    else:
        clean_output = generated
        
    return clean_output

def main():
    # 1. Caricamento Modello
    print("Caricamento Tokenizer e Modello...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Carica il modello base in float16
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # Carica l'adapter LoRA se esiste
    if os.path.exists(ADAPTER_PATH):
        print(f"Caricamento Adapter da {ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        model = model.merge_and_unload()
    else:
        print(f"⚠️ ATTENZIONE: Adapter non trovato in {ADAPTER_PATH}. Uso il modello base.")
    
    model.eval()

    # 2. Caricamento Test Set
    if not os.path.exists(DATASET_PATH):
        print(f"Errore: Dataset {DATASET_PATH} non trovato.")
        return

    dataset = load_from_disk(DATASET_PATH)
    test_data = dataset['test']
    
    RISK_MAP = {1: "Personal Harm", 2: "Physical Harm", 3: "Cybersecurity Harm"}
    
    predictions = []
    references_safe = []  # Ground Truth Safe
    references_action = [] # Action originale
    references_trigger = [] # Trigger originale
    
    print(f"Valutazione su {len(test_data)} campioni di test...")
    
    for i in tqdm(range(len(test_data))):
        row = test_data[i]
        
        desc = row['desc']
        just = row['justification']
        lbl = RISK_MAP.get(row['label'], "Unknown")
        
        # Generazione
        safe_model = generate_safe_rule(model, tokenizer, desc, just, lbl)
        predictions.append(safe_model)
        
        # Raccolta riferimenti
        references_safe.append(row['safe'])
        references_action.append(row['actionTitle'])
        references_trigger.append(row['triggerTitle'])

    # 3. Calcolo BERTScore
    print("Calcolo BERTScore in corso...")
    
    # F1 score per Safe Generato vs Safe Reale
    try:
        _, _, f1_safe = score(predictions, references_safe, lang="en", verbose=False)
        score_safe_pct = f1_safe.mean().item() * 100

        _, _, f1_action = score(predictions, references_action, lang="en", verbose=False)
        score_action_pct = f1_action.mean().item() * 100

        _, _, f1_trigger = score(predictions, references_trigger, lang="en", verbose=False)
        score_trigger_pct = f1_trigger.mean().item() * 100
    except Exception as e:
        print(f"Errore calcolo BERTScore: {e}")
        score_safe_pct = 0
        score_action_pct = 0
        score_trigger_pct = 0
        # garantisce variabili utilizzabili per i conteggi
        f1_safe = torch.zeros(len(predictions))

    # 4. Salvataggio Report
    output_txt = (
        f"=== REPORT VALUTAZIONE BERTScore ===\n"
        f"Campioni Test: {len(test_data)}\n"
        f"Modello: {ADAPTER_PATH}\n\n"
        f"1. Similarity (Generated Safe vs GT Safe): {score_safe_pct:.2f}%\n"
        f"2. Similarity (Action vs Generated Safe):  {score_action_pct:.2f}%\n"
        f"3. Similarity (Trigger vs Generated Safe): {score_trigger_pct:.2f}%\n"
    )
    
    print("\n" + output_txt)
    
    with open(EVAL_REPORT_FILE, "w") as f:
        f.write(output_txt)
    
    # --- DEBUG: per-sample BERTScore dump ---
    debug_file = "bertscore_pairs_debug.txt"
    with open(debug_file, "w", encoding="utf-8") as df:
        for i, (pred, ref_safe, ref_action, ref_trigger) in enumerate(zip(predictions, references_safe, references_action, references_trigger)):
            # Calcolo BERTScore per questo singolo campione
            try:
                _, _, f1_safe_single = score([pred], [ref_safe], lang="en", verbose=False)
                _, _, f1_action_single = score([pred], [ref_action], lang="en", verbose=False)
                _, _, f1_trigger_single = score([pred], [ref_trigger], lang="en", verbose=False)
                score_safe_single = f1_safe_single.item() * 100
                score_action_single = f1_action_single.item() * 100
                score_trigger_single = f1_trigger_single.item() * 100
            except Exception:
                score_safe_single = 0.0
                score_action_single = 0.0
                score_trigger_single = 0.0
            
            df.write(f"INDEX: {i}\n")
            df.write(f"REF_SAFE: {ref_safe}\n")
            df.write(f"PRED_SAFE: {pred}\n")
            df.write(f"REF_ACTION: {ref_action}\n")
            df.write(f"REF_TRIGGER: {ref_trigger}\n")
            df.write(f"BERTScore Safe vs PRED: {score_safe_single:.2f}%\n")
            df.write(f"BERTScore Action vs PRED: {score_action_single:.2f}%\n")
            df.write(f"BERTScore Trigger vs PRED: {score_trigger_single:.2f}%\n")
            df.write("---\n")
    print(f"Debug dump scritto in {debug_file}")
        
    # 5. Generazione Grafico Loss (Nuova funzione robusta)
    print("Generazione grafico Loss...")
    plot_training_loss(LOG_FILE, LOSS_PLOT_FILE)

if __name__ == "__main__":
    main()