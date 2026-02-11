import os
import torch
import nltk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_from_disk
from tqdm import tqdm
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

# --- CONFIGURAZIONE ---
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_PATH = "./output_llama_standard_vsenzaGRPO/final_adapter"
DATASET_PATH = "dataset_final_processed"
EVAL_REPORT_FILE = "eval_report_meteor.txt"

def ensure_nltk_resources():
    """Scarica le risorse necessarie per la tokenizzazione e METEOR."""
    resources = ['punkt', 'wordnet', 'omw-1.4', 'punkt_tab']
    for res in resources:
        nltk.download(res, quiet=True)

def generate_safe_rule(model, tokenizer, description, justification, label_str, max_new_tokens=128):
    user_msg = (
        f"DESCRIPTION: {description}\n"
        f"JUSTIFICATION: {justification}\n"
        f"LABEL: {label_str}"
    )

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
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Pulizia output: cerchiamo la parte dopo l'istruzione dell'assistente
    if "assistant" in generated:
        clean_output = generated.split("assistant")[-1].strip()
        # Rimuoviamo l'eventuale prefisso SAFE_MODEL: se presente nella generazione
        if "SAFE_MODEL:" in clean_output:
            clean_output = clean_output.split("SAFE_MODEL:")[-1].strip()
    else:
        clean_output = generated.strip()

    return clean_output

def main():
    ensure_nltk_resources()

    print("Caricamento Tokenizer e Modello...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    if os.path.exists(ADAPTER_PATH):
        print(f"Caricamento Adapter da {ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        model = model.merge_and_unload()
    else:
        print(f"⚠️ ATTENZIONE: Adapter non trovato. Uso il modello base.")

    model.eval()

    if not os.path.exists(DATASET_PATH):
        print(f"Errore: Dataset {DATASET_PATH} non trovato.")
        return

    dataset = load_from_disk(DATASET_PATH)
    test_data = dataset['test']

    predictions = []
    references_safe = []

    print(f"Valutazione su {len(test_data)} campioni...")
    per_sample_scores = []

    for i in tqdm(range(len(test_data))):
        row = test_data[i]
        pred_text = generate_safe_rule(model, tokenizer, row['desc'], row['justification'], str(row.get('label', '')))
        ref_text = row['safe']

        # --- CORREZIONE METEOR ---
        # Tokenizzazione necessaria: meteor_score vuole liste di parole
        pred_tokens = word_tokenize(pred_text)
        ref_tokens = word_tokenize(ref_text)
        
        try:
            # Calcolo basato su token
            score = meteor_score([ref_tokens], pred_tokens)
        except Exception as e:
            # Debug in caso di fallimento tecnico
            if i == 0: print(f"Errore tecnico METEOR all'indice {i}: {e}")
            score = 0.0
            
        per_sample_scores.append(score)
        predictions.append(pred_text)
        references_safe.append(ref_text)

    avg_meteor = (sum(per_sample_scores) / len(per_sample_scores)) * 100 if per_sample_scores else 0.0

    # Report Finale
    threshold = 0.5
    pct_above_thresh = (sum(1 for s in per_sample_scores if s >= threshold) / len(per_sample_scores)) * 100
    
    report = (
        f"=== REPORT VALUTAZIONE METEOR CORRETTO ===\n"
        f"Campioni Test: {len(test_data)}\n"
        f"METEOR Medio: {avg_meteor:.2f}%\n"
        f"Percentuale sopra soglia {threshold}: {pct_above_thresh:.2f}%\n"
    )

    print(report)
    with open(EVAL_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    # Debug File
    with open("meteor_pairs_debug.txt", "w", encoding="utf-8") as df:
        for i, (p, r, s) in enumerate(zip(predictions, references_safe, per_sample_scores)):
            df.write(f"INDEX: {i} | SCORE: {s:.4f}\nREF: {r}\nPRED: {p}\n---\n")

if __name__ == "__main__":
    main()