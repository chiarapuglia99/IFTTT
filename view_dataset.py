import os
from datasets import load_from_disk

def inspect_processed_data(dataset_path="dataset_llama"):
    if not os.path.exists(dataset_path):
        print(f"Errore: La cartella '{dataset_path}' non esiste. Esegui prima lo script di preparazione.")
        return

    # Carica il dataset dal disco
    dataset = load_from_disk(dataset_path)
    
    print("="*50)
    print(f"STRUTTURA DEL DATASET: {dataset}")
    print("="*50)

    # Prendiamo un esempio dal training set
    sample = dataset['train'][0]

    print("\n--- ESEMPIO DI PROMPT (Input per il modello) ---")
    print(sample['prompt'])
    
    print("\n--- ESEMPIO DI COMPLETION (Target per la Reward) ---")
    print(sample['completion'])
    
    print("\n--- METADATI PER LE REWARD FUNCTIONS ---")
    print(f"Target Category: {sample['target_category']}")
    print(f"Target Justification: {sample['target_justification']}")
    # Stampa i primi 100 caratteri della versione safe
    print(f"Target Safe (snippet): {sample['target_safe'][:100]}...")
    print("="*50)

    # Statistiche rapide
    print(f"\nNumero totale di esempi nel Train: {len(dataset['train'])}")
    print(f"Numero totale di esempi nel Validation: {len(dataset['validation'])}")
    print(f"Numero totale di esempi nel Test: {len(dataset['test'])}")

if __name__ == "__main__":
    inspect_processed_data()