import pandas as pd
from datasets import Dataset, DatasetDict
import os
from transformers import AutoTokenizer

# --- CONFIGURAZIONE ---
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
CSV_PATH = "dataset.csv"
OUTPUT_DIR = "dataset_final_processed_nobalance"
RISK_MAP = {1: "Personal Harm", 2: "Physical Harm", 3: "Cybersecurity Harm"}

def create_prompt(row):
    risk_class = RISK_MAP.get(row['label'], "Unknown Risk")
    
    # System Prompt ottimizzato per la sicurezza IoT
    system_msg = (
        "You are an expert AI security assistant specializing in IoT automation. "
        "Your task is to rewrite an unsafe automation rule into a safe variant based strictly on the provided security justification.\n"
        "GUIDELINES:\n"
        "1. STRUCTURE: Follow the standard 'IF [Trigger] THEN [Action]' format.\n"
        "2. PRESERVE INTENT: Keep the original action and devices unless they constitute the risk.\n"
        "3. APPLY SECURITY: Add specific conditions (e.g., user confirmation, location checks) derived from the justification.\n"
        "4. CONCISE OUTPUT: Generate ONLY the safe rule text."
    )
    
    user_msg = (
        "Task: Secure this Automation Rule\n"
        f"Input Rule: {row['desc']}\n"
        f"Risk Category: {risk_class}\n"
        f"Security Justification: {row['justification']}\n\n"
        "Output the Safe Rule Variant:"
    )
    
    assistant_msg = row['safe']

    # Formattazione Llama 3 Instruct
    text = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant_msg}<|eot_id|>"
    )
    return text

def main():
    print("1. Caricamento Dataset...")
    if not os.path.exists(CSV_PATH):
        print(f"Errore: {CSV_PATH} non trovato.")
        return

    df = pd.read_csv(CSV_PATH)
    
    # --- NO BILANCIAMENTO ---
    # Usiamo tutto il dataset (shuffle per mescolare le classi)
    print(f"Utilizzo dell'intero dataset: {len(df)} righe.")
    df_final = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    raw_dataset = Dataset.from_pandas(pd.DataFrame({"text": df_final.apply(create_prompt, axis=1)}))
    
    print("2. Tokenizzazione...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=1024, 
            padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    print("3. Split Train/Val/Test...")
    train_testvalid = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
    
    final_dataset = DatasetDict({
        'train': train_testvalid['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']
    })
    
    final_dataset.save_to_disk(OUTPUT_DIR)
    print("✅ Dataset pronto.")

if __name__ == "__main__":
    main()