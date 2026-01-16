import pandas as pd
from datasets import Dataset, DatasetDict
import os
from tqdm import tqdm

def prepare_data(csv_path, output_dir="processed_dataset"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print("[1/6] Caricamento dati...")
    df = pd.read_csv(csv_path)
    print(f"Dati caricati: {len(df)} righe")

    # 2. Mapping dei danni
    harm_mapping = {
        1: "Personal harm",
        2: "Physical harm",
        3: "Cybersecurity harm"
    }

    print("[2/6] Creazione prompt in inglese...")
    def create_prompt(row):
        instruction = (
            "Analyze the following IFTTT automation rule:\n"
            f"- Trigger: {row['triggerTitle']} (Channel: {row['triggerChannelTitle']})\n"
            f"- Action: {row['actionTitle']} (Channel: {row['actionChannelTitle']})\n"
            f"- Description: {row['desc']}\n\n"
            "Task:\n"
            "1. Identify the harm category (Personal harm, Physical harm, or Cybersecurity harm).\n"
            "2. Provide a justification of why this rule is risky.\n"
            "3. Propose a 'safe' version of the rule to mitigate the risk.\n\n"
            "Response Format:\n"
            "Harm Category: <category>\n"
            "Justification: <reason>\n"
            "Safe Version: <mitigation>"
        )
        return instruction

    tqdm.pandas(desc="Creazione prompt")
    df['prompt'] = df.progress_apply(create_prompt, axis=1)
    df['harm_type'] = df['label'].map(harm_mapping)
    print("Prompt creati.")

    print("[3/6] Selezione colonne utili...")
    dataset_df = df[['prompt', 'harm_type', 'justification', 'safe']]
    print("Colonne selezionate.")

    print("[4/6] Conversione in Hugging Face Dataset...")
    full_dataset = Dataset.from_pandas(dataset_df)
    print("Dataset Hugging Face creato.")

    print("[5/6] Suddivisione in train/val/test...")
    train_test_split = full_dataset.train_test_split(test_size=0.2, seed=42)
    temp_split = train_test_split['test'].train_test_split(test_size=0.75, seed=42)

    final_dataset = DatasetDict({
        'train': train_test_split['train'],
        'validation': temp_split['train'], # Questo è il 5%
        'test': temp_split['test']         # Questo è il 15%
    })
    print("Split completato.")

    print("[6/6] Salvataggio su disco...")
    final_dataset.save_to_disk(output_dir)
    print(f"Dataset salvato in: {output_dir}")
    print("-" * 30)
    print(f"Training set size:   {len(final_dataset['train'])} examples (80%)")
    print(f"Validation set size: {len(final_dataset['validation'])} examples (5%)")
    print(f"Test set size:       {len(final_dataset['test'])} examples (15%)")
    print("-" * 30)
    print("Example Prompt:")
    print(final_dataset['train'][0]['prompt'])

if __name__ == "__main__":
    prepare_data("dataset.csv")