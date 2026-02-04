import pandas as pd
from datasets import Dataset, DatasetDict
import os
from transformers import AutoTokenizer

# --- CONFIGURAZIONE ---
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
CSV_PATH = "dataset.csv"
OUTPUT_DIR = "dataset_final_processed"
RISK_MAP = {1: "Personal Harm", 2: "Physical Harm", 3: "Cybersecurity Harm"}

def create_prompt(row):
    risk_class = RISK_MAP.get(row['label'], "Unknown Risk")
    
    # SYSTEM PROMPT (ENGLISH & OPTIMIZED)
    # Strutturato con direttive chiare per massimizzare la compliance del modello
    system_msg = (
        "You are an expert AI security assistant specializing in IoT automation. "
        "Your task is to rewrite an unsafe automation rule into a safe variant based strictly on the provided security justification.\n"
        "GUIDELINES:\n"
        "1. PRESERVE INTENT: Keep the original action and devices. Do not change the core functionality unless it constitutes the risk.\n"
        "2. APPLY SECURITY: Add specific conditions (e.g., user confirmation, location checks, trusted senders) derived directly from the 'Security Justification'.\n"
        "3. NO HALLUCINATIONS: Do not invent devices, brands (e.g., WeMo, Nest), or locations not explicitly mentioned in the input.\n"
        "4. CONCISE OUTPUT: Generate ONLY the safe rule text. Do not provide explanations or chat."
    )
    
    # USER INPUT (Structured for clarity)
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
    
    # --- BILANCIAMENTO ---
    df_1 = df[df['label'] == 1]
    df_2 = df[df['label'] == 2]
    df_3 = df[df['label'] == 3]
    
    min_size = min(len(df_1), len(df_2), len(df_3)) # 727
    print(f"Bilanciamento a {min_size} campioni per classe.")
    
    df_bal = pd.concat([
        df_1.sample(min_size, replace=False, random_state=42),
        df_2.sample(min_size, replace=False, random_state=42),
        df_3.sample(min_size, replace=False, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Creazione dataset grezzo
    raw_dataset = Dataset.from_pandas(pd.DataFrame({"text": df_bal.apply(create_prompt, axis=1)}))
    
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
    
    print("3. Split Train (80%) / Validation (10%) / Test (10%)...")
    train_testvalid = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
    
    final_dataset = DatasetDict({
        'train': train_testvalid['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']
    })
    
    print(f"Dimensioni: Train={len(final_dataset['train'])}, Val={len(final_dataset['validation'])}, Test={len(final_dataset['test'])}")
    final_dataset.save_to_disk(OUTPUT_DIR)
    print("✅ Dataset pronto.")

if __name__ == "__main__":
    main()