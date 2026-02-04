import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value
import os
from tqdm import tqdm

def prepare_data_safe_only(csv_path, output_dir="dataset_qwen_safe_only_v3"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print("[1/2] Caricamento e Bilanciamento dati...")
    
    df = pd.read_csv(csv_path)
    df = df.fillna("") 

    harm_mapping = {1: "personal harm", 2: "physical harm", 3: "cybersecurity harm"}
    df['harm_type_str'] = df['label'].map(harm_mapping).fillna("unknown risk")

    min_size = df['label'].value_counts().min()
    print(f"Bilanciamento a {min_size} elementi per classe...")

    df_balanced = df.groupby('label').apply(
        lambda x: x.sample(min_size, random_state=42)
    ).reset_index(drop=True)

    # --- NUOVI FEW-SHOT EXAMPLES (Più vari e concisi) ---
    # Abbiamo tolto il Geofencing ripetitivo e messo logiche diverse:
    # 1. Whitelist (Cyber)
    # 2. Privacy Setting (Personal)
    # 3. Encryption/Password (Physical/Generic)
    few_shot_examples = """
EXAMPLE 1 (Logic: Whitelist/Trusted Source):
Input:
- Risky Rule: Any new email -> Save attachment to Drive
- Identified Risk: cybersecurity harm
- Reasoning: Malicious attachments could be saved automatically.
Output: <safe_rule>Only save attachments to Drive if the sender is in your 'Trusted Contacts' list to prevent malware storage.</safe_rule>

EXAMPLE 2 (Logic: Privacy Restrictions):
Input:
- Risky Rule: New photo -> Post to Facebook
- Identified Risk: personal harm
- Reasoning: Private photos might be shared publicly.
Output: <safe_rule>Post new photos to Facebook but set the default privacy visibility to 'Only Me' or 'Friends' to prevent public exposure.</safe_rule>

EXAMPLE 3 (Logic: Human Verification):
Input:
- Risky Rule: Voice command -> Unlock front door
- Identified Risk: physical harm
- Reasoning: An unauthorized person could shout the command to enter.
Output: <safe_rule>Unlock the door via voice command only if a secondary PIN code is provided or a specific voice match is verified.</safe_rule>
"""

    def create_structure(row):
        # SYSTEM PROMPT AGGIORNATO:
        # Aggiunto vincolo forte sulla lunghezza: "Keep the rule concise (max 1-2 sentences)"
        system_msg = (
            "You are a security expert specialized in IoT automation mitigation.\n"
            "Your task is to provide a SAFER ALTERNATIVE (Safe Rule) for a risky automation rule.\n"
            "Propose a concrete technical solution that mitigates the risk (e.g. adding conditions, filters, or restrictions).\n"
            "IMPORTANT: Keep the rule concise (max 30 words). Do not explain, just state the rule.\n\n"
            f"Examples:\n{few_shot_examples}\n\n"
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
            "target": f"<safe_rule>{row['safe']}</safe_rule><|im_end|>",
            "original_trigger": row['triggerTitle'],
            "original_action": row['actionTitle']
        }

    print("[2/2] Creazione dataset HuggingFace...")
    tqdm.pandas()
    processed_data = df_balanced.apply(create_structure, axis=1).tolist()
    dataset_df = pd.DataFrame(processed_data)

    features = Features({
        'prompt': Value('string'), 
        'target': Value('string'),
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