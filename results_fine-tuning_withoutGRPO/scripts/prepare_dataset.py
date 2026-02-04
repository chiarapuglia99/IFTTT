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
    description = row['desc']
    justification = row['justification']
    label_val = row['label']
    risk_class = RISK_MAP.get(label_val, "Unknown Risk")
    target_safe = row['safe']
    
    system_msg = (
        "You are an expert AI security assistant. "
        "Your task is to generate a 'safe' automation rule based on the description, justification, and risk category provided.\n"
        "Input Format:\n"
        "DESCRIPTION: <description of the rule>\n"
        "JUSTIFICATION: <security justification>\n"
        "LABEL: <risk category>\n\n"
        "Output:\n"
        "SAFE_MODEL: <safe automation rule>"
    )
    
    user_msg = (
        f"DESCRIPTION: {description}\n"
        f"JUSTIFICATION: {justification}\n"
        f"LABEL: {risk_class}"
    )
    
    full_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_msg}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{target_safe}<|eot_id|>"
    )
    
    return full_prompt

def main():
    print("1. Caricamento dataset...")
    df = pd.read_csv(CSV_PATH)
    print(f"Dataset caricato: {len(df)} righe.")
    
    raw_dataset = Dataset.from_pandas(df)
    raw_dataset = raw_dataset.map(lambda x: {"text": create_prompt(x)})

    print("2. Tokenizzazione...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=512, # RIDOTTO da 1024 a 512 per salvare memoria sulla 1050 Ti
            padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
    
    print("3. Split Train (70%) / Validation (10%) / Test (20%)...")
    train_temp = tokenized_dataset.train_test_split(test_size=0.3, seed=42)
    test_valid = train_temp['test'].train_test_split(test_size=0.6666, seed=42)
    
    final_dataset = DatasetDict({
        'train': train_temp['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']
    })
    
    print(f"Dimensioni finali:\nTrain: {len(final_dataset['train'])}\nValidation: {len(final_dataset['validation'])}\nTest: {len(final_dataset['test'])}")
    final_dataset.save_to_disk(OUTPUT_DIR)
    print(f"Dataset salvato in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()