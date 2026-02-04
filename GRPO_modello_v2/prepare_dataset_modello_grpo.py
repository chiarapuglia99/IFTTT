import pandas as pd
from sklearn.model_selection import train_test_split
import os
from transformers import AutoTokenizer

# --- CONFIGURAZIONE ---
BASE_DIR = 'dataset_modello_grpo'
INPUT_FILE = os.path.join(BASE_DIR, 'dataset.csv')
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct" # Il modello esatto per caricare il template giusto

# Mappa Rischi
RISK_MAP = {1: "Privacy Harm", 2: "Physical Safety Harm", 3: "Cybersecurity Harm"}

def prepare_dataset_jsonl():
    print(f"--- STEP 1: Preparazione Dataset Qwen-Native ---")
    
    # 1. Carichiamo il Tokenizer (Serve per formattare correttamente i tag <|im_start|>)
    print(f"Caricamento Tokenizer da {MODEL_ID}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"Errore caricamento tokenizer: {e}")
        return

    # 2. Caricamento Dati
    if not os.path.exists(INPUT_FILE):
        if os.path.exists('dataset.csv'):
            print(f"File non trovato in {BASE_DIR}, carico dalla root.")
            df = pd.read_csv('dataset.csv')
        else:
            raise FileNotFoundError("Dataset non trovato!")
    else:
        df = pd.read_csv(INPUT_FILE)

    print(f"Dataset caricato: {len(df)} righe.")
    os.makedirs(BASE_DIR, exist_ok=True)

    # 3. Funzione che costruisce il Prompt Formattato (Chat Template)
    def create_chat_prompt(row):
        risk_class = RISK_MAP.get(row['label'], "General Safety Risk")
        
        # System Prompt (Istruzioni + Linee Guida)
        system_content = (
            "You are an expert AI security assistant. Your task is to rewrite unsafe IoT rules into safe, natural sentences.\n"
            "GUIDELINES:\n"
            "1. NATURAL PHRASING: Use connectives like 'but only if', 'provided that'.\n"
            "2. PRESERVE TRIGGER: Keep the original 'when' condition.\n"
            "3. NO HALLUCINATIONS: Do not invent PIN codes or devices not present in justification.\n"
            "4. FORMAT: Output MUST be enclosed in <safe_rule> tags."
        )

        # Few-Shot Examples (Inseriti come contesto utente/assistente precedenti per rafforzare l'apprendimento)
        # Nota: Qwen gestisce bene il contesto nella history.
        
        # Costruiamo la history dei messaggi
        messages = [
            {"role": "system", "content": system_content},
            
            # Esempio 1 (Privacy)
            {"role": "user", "content": "Rule: Share photos to Facebook when taken.\nRisk: Privacy.\nJustification: Prevent sharing private photos."},
            {"role": "assistant", "content": "<safe_rule>Share photos to Facebook when taken, provided they are explicitly tagged '#Public'.</safe_rule>"},
            
            # Esempio 2 (Safety)
            {"role": "user", "content": "Rule: Turn on the heater when I leave work.\nRisk: Physical Safety.\nJustification: Fire hazard if house empty."},
            {"role": "assistant", "content": "<safe_rule>Turn on the heater when you leave work, but only if the house is currently occupied.</safe_rule>"},
            
            # Il Task Corrente
            {"role": "user", "content": (
                f"Task: Secure this Rule\n"
                f"Input Rule: {row['desc']}\n"
                f"Risk Category: {risk_class}\n"
                f"Security Justification: {row['justification']}\n\n"
                "Output the Safe Rule Variant:"
            )}
        ]
        
        # APPLICAZIONE TEMPLATE QWEN
        # Questo trasforma la lista in: <|im_start|>system...<|im_end|><|im_start|>user...
        # add_generation_prompt=True aggiunge <|im_start|>assistant alla fine, pronto per la risposta.
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print("Applicazione del Chat Template (può richiedere qualche secondo)...")
    df['prompt'] = df.apply(create_chat_prompt, axis=1)

    # 4. Creazione Target (Solo XML)
    def create_xml_target(row):
        safe_text = str(row['safe']).strip()
        safe_text = safe_text.replace("<safe_rule>", "").replace("</safe_rule>", "")
        return f"<safe_rule>{safe_text}</safe_rule>"

    df['completion'] = df.apply(create_xml_target, axis=1)

    # 5. Export
    output_columns = ['prompt', 'completion', 'desc', 'label', 'justification']
    df_clean = df[output_columns]

    train, temp = train_test_split(df_clean, test_size=0.2, random_state=42, stratify=df_clean['label'])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['label'])

    train.to_json(os.path.join(BASE_DIR, 'train.jsonl'), orient='records', lines=True)
    val.to_json(os.path.join(BASE_DIR, 'val.jsonl'), orient='records', lines=True)
    test.to_json(os.path.join(BASE_DIR, 'test.jsonl'), orient='records', lines=True)

    print("\nDataset Qwen-Ready generato!")
    print(f"Esempio Prompt RAW (visibile al modello):\n{train.iloc[0]['prompt'][:200]}...")
    
if __name__ == "__main__":
    prepare_dataset_jsonl()