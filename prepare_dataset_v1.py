import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value
import os
from tqdm import tqdm

def prepare_data_for_llama_grpo(csv_path, output_dir="dataset_llama4"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print("[1/3] Caricamento e Bilanciamento dati...")
    df = pd.read_csv(csv_path)
    
    harm_mapping = {
        1: "personal harm",
        2: "physical harm",
        3: "cybersecurity harm"
    }
    df['harm_type_str'] = df['label'].map(harm_mapping)

    # Esempi Few-Shot aggiornati con l'incipit obbligatorio
    example_cyber = df[df['label'] == 3].iloc[0]
    example_physical = df[df['label'] == 2].iloc[0]
    example_personal = df[df['label'] == 1].iloc[0]

    few_shot_context = (
        f"EXAMPLES OF CORRECT RESPONSES:\n\n"
        f"Example 1:\n<category>cybersecurity harm</category><justification>This rule might cause a cybersecurity harm because {example_cyber['justification'][example_cyber['justification'].lower().find('because')+8:] if 'because' in example_cyber['justification'].lower() else example_cyber['justification']}</justification><safe_rule>{example_cyber['safe']}</safe_rule>\n\n"
        f"Example 2:\n<category>physical harm</category><justification>This rule might cause a physical harm because {example_physical['justification'][example_physical['justification'].lower().find('because')+8:] if 'because' in example_physical['justification'].lower() else example_physical['justification']}</justification><safe_rule>{example_physical['safe']}</safe_rule>\n\n"
        f"Example 3:\n<category>personal harm</category><justification>This rule might cause a personal harm because {example_personal['justification'][example_personal['justification'].lower().find('because')+8:] if 'because' in example_personal['justification'].lower() else example_personal['justification']}</justification><safe_rule>{example_personal['safe']}</safe_rule>\n"
    )

    min_size = df['label'].value_counts().min()
    df_balanced = df.groupby('label').apply(
        lambda x: x.sample(min_size, random_state=42)
    ).reset_index(drop=True)

    def create_grpo_structure(row):
        # SYSTEM PROMPT: Integrato con incipit obbligatorio e vincoli sintattici
        system_msg = (
            "You are a strict security auditor. Respond ONLY with XML tags.\n"
            "RULES:\n"
            "1. You MUST include all three tags: <category>, <justification>, <safe_rule>.\n"
            "2. CATEGORY MUST BE: 'personal harm', 'physical harm', or 'cybersecurity harm'.\n"
            "3. The <justification> MUST start exactly with 'This rule might cause a [Category] harm because...' and explain the technical attack chain.\n"
            "4. Every tag MUST have a matching opening and closing tag.\n"
            "5. Be analytical and detailed. Do not be overly brief.\n"
            "6. NO introductions, NO conversational text, NO notes. Stop immediately after </safe_rule>.\n\n"
            f"{few_shot_context}"
        )
        
        user_msg = (
            f"Analyze this rule:\n"
            f"Trigger: {row['triggerTitle']} ({row['triggerChannelTitle']})\n"
            f"Action: {row['actionTitle']} ({row['actionChannelTitle']})\n"
            f"Description: {row['desc']}"
        )

        full_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        target_response = (
            f"<category>{row['harm_type_str']}</category>\n"
            f"<justification>{row['justification']}</justification>\n"
            f"<safe_rule>{row['safe']}</safe_rule><|eot_id|>"
        )
        
        return {
            "prompt": full_prompt,
            "completion": target_response,
            "target_category": row['harm_type_str'],
            "target_justification": row['justification'],
            "target_safe": row['safe']
        }

    print("[2/3] Generazione prompt con incipit forzato e vincoli XML...")
    tqdm.pandas()
    processed_data = df_balanced.apply(create_grpo_structure, axis=1).tolist()
    dataset_df = pd.DataFrame(processed_data)

    features = Features({
        'prompt': Value('string'),
        'completion': Value('string'),
        'target_category': Value('string'),
        'target_justification': Value('string'),
        'target_safe': Value('string')
    })

    full_dataset = Dataset.from_pandas(dataset_df, features=features)
    splits = full_dataset.train_test_split(test_size=0.2, seed=42)
    test_val = splits['test'].train_test_split(test_size=0.5, seed=42)

    final_ds = DatasetDict({'train': splits['train'], 'validation': test_val['train'], 'test': test_val['test']})
    final_ds.save_to_disk(output_dir)
    print(f"Dataset salvato in: {output_dir}")

if __name__ == "__main__":
    prepare_data_for_llama_grpo("dataset.csv")