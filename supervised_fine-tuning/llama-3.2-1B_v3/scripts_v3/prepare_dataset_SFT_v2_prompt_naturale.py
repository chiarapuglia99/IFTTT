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
    
    # --- MODIFICA: PROMPT PER LINGUAGGIO NATURALE MA LOGICA FERREA ---
    system_msg = (
        "You are an expert AI security assistant. Your task is to rewrite unsafe IoT rules into safe, natural sentences.\n"
        "GUIDELINES:\n"
        "1. NATURAL PHRASING: Do not use rigid 'IF-THEN' structures. Use natural connectives like 'but only if', 'provided that', 'requiring', or 'instead of'.\n"
        "2. PRESERVE TRIGGER: You must keep the original 'when' condition (e.g., 'when I leave'). Do NOT change the trigger time or location unless the risk explicitly demands it.\n"
        "3. NO HALLUCINATIONS: Do NOT add 'PIN codes', 'Verified Contacts', or 'Time Ranges' unless the Security Justification explicitly mentions them.\n"
        "4. LOGIC: Combine the [Original Trigger] with the [Security Constraint] seamlessly.\n"
    )
    
    # Esempi (Few-Shot) in LINGUAGGIO NATURALE
    few_shot_examples = (
        "EXAMPLE 1:\n"
        "Input Rule: If motion detected, unlock the door.\n"
        "Justification: Unlocking doors on motion is a security risk; require user approval.\n"
        "Safe Rule: Send a notification asking to unlock the door when motion is detected.\n\n"
        
        "EXAMPLE 2:\n"
        "Input Rule: Turn on the heater when I leave work.\n"
        "Justification: Prevent fire hazard if the house is empty for too long.\n"
        "Safe Rule: Turn on the heater when you leave work, but only if the house is currently occupied.\n\n"
        
        "EXAMPLE 3 (Anti-Hallucination):\n"
        "Input Rule: Share photos to Facebook when taken.\n"
        "Justification: Prevent sharing private photos; restrict to specific tag.\n"
        "Safe Rule: Share photos to Facebook when taken, provided they are explicitly tagged '#Public'.\n"
        "(Note: Do NOT invent 'verified friends' or 'PIN codes' here).\n\n"
    )

    user_msg = (
        f"{few_shot_examples}"
        "Task: Secure this Automation Rule\n"
        f"Input Rule: {row['desc']}\n"
        f"Risk Category: {risk_class}\n"
        f"Security Justification: {row['justification']}\n\n"
        "Output the Safe Rule Variant:"
    )
    
    assistant_msg = row['safe']

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