import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json

# ================= CONFIGURAZIONE GRPO =================
INPUT_FILE = 'dataset_80k.csv'
OUTPUT_DIR = 'processed_grpo_clean' # Cartella dedicata GRPO
RANDOM_SEED = 42

LABEL_MAPPING = {1: "Personal harm", 2: "Physical harm", 3: "Cybersecurity harm"}

# === SYSTEM PROMPT PULITO ===
SYSTEM_PROMPT = (
    "You are an expert in IoT security and safety. "
    "Your task is to analyze an automation rule and its associated risk category. "
    "You must provide a concise justification for why the rule is considered unsafe, "
    "and generate a safer variant of the rule.\n\n"
    "Your output must strictly follow this XML format:\n"
    "<justification> ...analysis of the risk... </justification>\n"
    "<safe> ...safer rule variant... </safe>"
)

def create_grpo_prompt(row):
    """Crea solo il prompt (System + User) per GRPO"""
    try: 
        label_val = int(row['target'])
    except: 
        label_val = 0
    
    description = str(row['desc']).strip()
    if not description or description.lower() == 'nan': 
        return None
    
    category = LABEL_MAPPING.get(label_val, "Unknown harm")
    
    # Prompt formattato per Qwen (senza risposta assistant)
    prompt_text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nDescription: {description}\nRisk Category: {category}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    return {"prompt": prompt_text}

def save_jsonl(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')

def main():
    print("🧹 Generazione Dataset GRPO Pulito V2 (Senza Few-Shot)...")
    if not os.path.exists(INPUT_FILE):
        print(f"Manca {INPUT_FILE}!")
        return

    # Gestione encoding per il file da 80k
    try:
        df = pd.read_csv(INPUT_FILE, sep=';', encoding='utf-8', on_bad_lines='skip')
    except:
        df = pd.read_csv(INPUT_FILE, sep=';', encoding='latin-1', on_bad_lines='skip')
    
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['desc', 'target'])
    
    processed_data = [x for x in df.apply(create_grpo_prompt, axis=1).tolist() if x]

    # Split: creiamo un set di addestramento e uno di validazione (eval)
    train, temp = train_test_split(processed_data, test_size=0.2, random_state=RANDOM_SEED)
    eval_set, _ = train_test_split(temp, test_size=0.5, random_state=RANDOM_SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_jsonl(train, os.path.join(OUTPUT_DIR, 'train_grpo.jsonl'))
    
    # SALVATAGGIO CON NOME CORRETTO: eval_grpo.jsonl
    save_jsonl(eval_set, os.path.join(OUTPUT_DIR, 'eval_grpo.jsonl'))
    
    print(f"✅ GRPO Dataset pronto in: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()