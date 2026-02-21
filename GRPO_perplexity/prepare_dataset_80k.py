import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json

# ================= CONFIGURAZIONE GRPO =================
INPUT_FILE = 'dataset_80k.csv'
OUTPUT_DIR = 'processed_grpo_clean' # Cartella dedicata GRPO
RANDOM_SEED = 42

LABEL_MAPPING = {1: "Personal harm", 2: "Physical harm", 3: "Cybersecurity harm"}

# === SYSTEM PROMPT FEW-SHOT (MIGLIORATO) ===
# Nessun input fittizio. Mostriamo SOLO come deve essere l'output dell'Assistant.
SYSTEM_PROMPT = (
    "You are an expert in IoT security and safety. "
    "Analyze the provided automation rule and its associated risk category. "
    "Output your reasoning strictly inside <justification> tags. "
    "Then, output a safer version of the rule strictly inside <safe> tags.\n\n"
    "--- EXAMPLE EXPECTED OUTPUT ---\n"
    "<justification> This rule is dangerous because an intruder or a stray animal could trigger the motion sensor, causing the front door to unlock and granting unauthorized access to the house. </justification>\n"
    "<safe> If motion is detected outside, turn on the porch light and send a notification to the user's phone, but do not automatically unlock the door. </safe>\n"
    "-------------------------------"
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
    print("🧹 Generazione Dataset GRPO Pulito V2...")
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Manca {INPUT_FILE}!")
        return

    # Gestione encoding per il file da 80k
    try:
        df = pd.read_csv(INPUT_FILE, sep=';', encoding='utf-8', on_bad_lines='skip')
    except:
        df = pd.read_csv(INPUT_FILE, sep=';', encoding='latin-1', on_bad_lines='skip')
    
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['desc', 'target'])
    
    processed_data = [x for x in df.apply(create_grpo_prompt, axis=1).tolist() if x]

    # Split (usiamo train e test)
    train, temp = train_test_split(processed_data, test_size=0.2, random_state=RANDOM_SEED)
    val, test = train_test_split(temp, test_size=0.5, random_state=RANDOM_SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_jsonl(train, os.path.join(OUTPUT_DIR, 'train_grpo.jsonl'))
    save_jsonl(test, os.path.join(OUTPUT_DIR, 'test_grpo.jsonl'))
    
    print(f"✅ GRPO Dataset pronto in: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()