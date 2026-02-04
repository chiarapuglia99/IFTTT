import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value
import os
from tqdm import tqdm

def prepare_data_safe_only(csv_path, output_dir="dataset_qwen_safe_only"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print("[1/2] Caricamento e Bilanciamento dati...")
    
    df = pd.read_csv(csv_path)
    df = df.fillna("") # Gestione valori nulli

    # Mappatura etichette
    harm_mapping = {1: "personal harm", 2: "physical harm", 3: "cybersecurity harm"}
    df['harm_type_str'] = df['label'].map(harm_mapping).fillna("unknown risk")

    print("\n--- PRIMA DEL BILANCIAMENTO ---")
    print(df['label'].value_counts())

    # --- LOGICA DI BILANCIAMENTO ATTIVA ---
    # Calcola la dimensione della classe più piccola
    min_size = df['label'].value_counts().min()
    print(f"\nLa classe più piccola ha {min_size} elementi. Bilancio tutto a {min_size}...")

    # Campiona 'min_size' elementi da ogni gruppo (label)
    df_balanced = df.groupby('label').apply(
        lambda x: x.sample(min_size, random_state=42)
    ).reset_index(drop=True)

    print("\n--- DOPO IL BILANCIAMENTO ---")
    print(df_balanced['label'].value_counts())
    print("--------------------------------\n")
    # --------------------------------------

    def create_structure(row):
        # SYSTEM: Ruolo esperto mitigazione
        system_msg = (
            "You are a security expert specialized in IoT automation.\n"
            "Your task is to provide a SAFER ALTERNATIVE (Safe Rule) for a risky automation rule.\n"
            "You will receive the rule description, the risk category, and the justification.\n"
            "Respond ONLY with the safe rule wrapped in <safe_rule> tags."
        )
        
        # USER: Input completo (Descrizione + Categoria + Giustificazione)
        user_msg = (
            f"Risky Rule: {row['triggerTitle']} -> {row['actionTitle']}\n"
            f"Description: {row['desc']}\n"
            f"Identified Risk: {row['harm_type_str']}\n"
            f"Reasoning: {row['justification']}\n\n"
            "Task: Write a safe rule that mitigates this specific risk by modifying the Trigger or the Action (e.g., adding conditions, filters, or rate limits)."
        )

        full_prompt = (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        # COMPLETION: Solo Safe Rule
        return {
            "prompt": full_prompt,
            "completion": f"<safe_rule>{row['safe']}</safe_rule><|im_end|>",
            # Campi extra per la reward function Trigger/Action
            "original_trigger": row['triggerTitle'],
            "original_action": row['actionTitle']
        }

    print("[2/2] Creazione dataset HuggingFace...")
    tqdm.pandas()
    processed_data = df_balanced.apply(create_structure, axis=1).tolist()
    dataset_df = pd.DataFrame(processed_data)

    features = Features({
        'prompt': Value('string'), 
        'completion': Value('string'),
        'original_trigger': Value('string'),
        'original_action': Value('string')
    })
    
    full_dataset = Dataset.from_pandas(dataset_df, features=features)
    splits = full_dataset.train_test_split(test_size=0.1, seed=42)

    final_ds = DatasetDict({'train': splits['train'], 'test': splits['test']})
    final_ds.save_to_disk(output_dir)
    print(f"✅ Dataset salvato in: {output_dir}")

if __name__ == "__main__":
    prepare_data_safe_only("dataset.csv")