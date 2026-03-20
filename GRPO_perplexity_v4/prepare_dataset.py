import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json

# ================= CONFIGURAZIONE SFT =================
INPUT_FILE = 'dataset.csv'
OUTPUT_DIR = 'processed_sft_clean' # Cartella dedicata SFT
RANDOM_SEED = 42

# Mapping delle label
LABEL_MAPPING = {1: "Personal harm", 2: "Physical harm", 3: "Cybersecurity harm"}

# === SYSTEM PROMPT "PULITO" (Senza esempi ingannevoli) ===
SYSTEM_PROMPT = (
    "You are an expert in IoT security and safety. "
    "Your task is to analyze an automation rule and its associated risk category. "
    "You must provide a concise justification for why the rule is considered unsafe, "
    "and generate a safer variant of the rule.\n\n"
    "Your output must strictly follow this XML format:\n"
    "<justification> ...analysis of the risk... </justification>\n"
    "<safe> ...safer rule variant... </safe>"
)

def create_sft_conversation(row):
    """Crea input (User) + output (Assistant) per SFT"""
    description = row['desc']
    category = LABEL_MAPPING.get(row['label'], "Unknown harm")
    
    # User Input
    user_content = f"Description: {description}\nRisk Category: {category}"
    
    # Assistant Output (Target) - Questo lo impara l'SFT
    assistant_content = (
        f"<justification>{row['justification']}</justification>\n"
        f"<safe>{row['safe']}</safe>"
    )
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }

def save_jsonl(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f); f.write('\n')

def main():
    print("🧹 Generazione Dataset SFT Pulito...")
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Manca {INPUT_FILE}!"); return

    df = pd.read_csv(INPUT_FILE)
    processed_data = df.apply(create_sft_conversation, axis=1).tolist()

    # Split 80/10/10
    train, temp = train_test_split(processed_data, test_size=0.2, random_state=RANDOM_SEED)
    val, test = train_test_split(temp, test_size=0.5, random_state=RANDOM_SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_jsonl(train, os.path.join(OUTPUT_DIR, 'train.jsonl'))
    save_jsonl(val, os.path.join(OUTPUT_DIR, 'validation.jsonl'))
    save_jsonl(test, os.path.join(OUTPUT_DIR, 'test.jsonl'))
    
    print(f"✅ SFT Dataset pronto in: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()