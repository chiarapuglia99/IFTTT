import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value
import os
from tqdm import tqdm

def prepare_data_safe_only(csv_path, output_dir="dataset_qwen_safe_only_v2"): # Ho cambiato nome cartella per sicurezza
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print("[1/2] Caricamento e Bilanciamento dati...")
    
    df = pd.read_csv(csv_path)
    df = df.fillna("") 

    # Mappatura etichette
    harm_mapping = {1: "personal harm", 2: "physical harm", 3: "cybersecurity harm"}
    df['harm_type_str'] = df['label'].map(harm_mapping).fillna("unknown risk")

    # Bilanciamento
    min_size = df['label'].value_counts().min()
    print(f"Bilanciamento a {min_size} elementi per classe...")

    df_balanced = df.groupby('label').apply(
        lambda x: x.sample(min_size, random_state=42)
    ).reset_index(drop=True)

    # --- FEW-SHOT EXAMPLES ---
    few_shot_examples = """
EXAMPLE 1 (Risk: Spam/Flooding -> Solution: Frequency Control):
Input:
- Risky Rule: Any new SMS sent -> Send a notification
- Identified Risk: cybersecurity harm
- Reasoning: An attacker could use a malicious app to send many sms, leading to spam notifications.
Output: <safe_rule>Trigger the notification when an SMS is sent, but apply a rate limit (e.g., once every 10 mins) to prevent spam flooding.</safe_rule>

EXAMPLE 2 (Risk: Physical Signaling -> Solution: Context Awareness):
Input:
- Risky Rule: Every day of the week at -> Turn off lights
- Identified Risk: physical harm
- Reasoning: Turning off lights in a predictable way signals that the house is empty, aiding attackers.
Output: <safe_rule>Turn off lights based on your actual location (leaving home) rather than a fixed time to avoid signaling an empty house.</safe_rule>

EXAMPLE 3 (Risk: Accidental Trigger -> Solution: Verification):
Input:
- Risky Rule: Device Disconnects -> Turn off switch
- Identified Risk: personal harm
- Reasoning: Accidentally deactivating Wifi could switch off devices even if the user is still home.
Output: <safe_rule>Turn off the D-Link switch when disconnecting from WiFi, but add a location check (Geofencing) to verify you actually left home.</safe_rule>
"""

    def create_structure(row):
        system_msg = (
            "You are a security expert specialized in IoT automation mitigation.\n"
            "Your task is to provide a SAFER ALTERNATIVE (Safe Rule) for a risky automation rule.\n"
            "Analyze the risky rule, the specific risk category, and the reasoning provided.\n"
            "Propose a concrete technical solution that mitigates the risk (e.g. by adding conditions, filters, permissions, or restrictions) without breaking the intended functionality.\n\n"
            f"Here are 3 examples of how to perform this task:\n{few_shot_examples}\n\n"
            "Now, provide the safe rule for the following case.\n"
            "Respond ONLY with the safe rule wrapped in <safe_rule> tags."
        )
        
        user_msg = (
            f"Input:\n"
            f"- Risky Rule: {row['triggerTitle']} -> {row['actionTitle']}\n"
            f"- Description: {row['desc']}\n"
            f"- Identified Risk: {row['harm_type_str']}\n"
            f"- Reasoning: {row['justification']}\n\n"
            "Output:"
        )

        full_prompt = (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        return {
            "prompt": full_prompt,
            # --- MODIFICA QUI: CAMBIATO NOME DA 'completion' A 'target' ---
            "target": f"<safe_rule>{row['safe']}</safe_rule><|im_end|>",
            # -------------------------------------------------------------
            "original_trigger": row['triggerTitle'],
            "original_action": row['actionTitle']
        }

    print("[2/2] Creazione dataset HuggingFace...")
    tqdm.pandas()
    processed_data = df_balanced.apply(create_structure, axis=1).tolist()
    dataset_df = pd.DataFrame(processed_data)

    # Aggiorniamo anche le Features col nuovo nome
    features = Features({
        'prompt': Value('string'), 
        'target': Value('string'), # <--- QUI
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