import os
import torch
import sys
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig

# --- LOGGER ---
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("training_log_reward.txt", "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()

# --- CONFIGURAZIONE ---
MODEL_NAME = "distilroberta-base" 
DATA_DIR = "dataset_modello_grpo"
OUTPUT_DIR = "reward_model_output"

def train_reward_model():
    print(f"--- Inizio Training Reward Model (Il Giudice Severo) ---")
    
    # 1. Caricamento Dataset
    data_files = {
        "train": os.path.join(DATA_DIR, "train.jsonl"),
        "validation": os.path.join(DATA_DIR, "val.jsonl")
    }
    dataset = load_dataset("json", data_files=data_files)

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.truncation_side = "left" # Se è troppo lungo, taglia l'inizio (istruzioni), non la fine (regola)

    # 3. Formattazione Dati "Anti-Imbroglio"
    print("Formattazione: Avvolgiamo ANCHE la risposta sbagliata nei tag per forzare il confronto sul contenuto...")
    
    def format_func(example):
        prompt = example.get('prompt', '')
        
        # Chosen (Safe): Ha già i tag dal dataset step 1
        # Es: <safe_rule>Turn on lights only if...</safe_rule>
        safe_response = example.get('completion', '')
        
        # Rejected (Risky): Originale. MA ORA AGGIUNGIAMO I TAG MANUALMENTE!
        # Così il modello non può dire "non ha i tag, quindi fa schifo". 
        # Deve leggere che il contenuto è pericoloso.
        risky_raw = example.get('desc', '').strip()
        risky_response = f"<safe_rule>{risky_raw}</safe_rule>"

        return {
            "chosen": f"{prompt}\n{safe_response}",
            "rejected": f"{prompt}\n{risky_response}"
        }

    original_columns = dataset["train"].column_names
    dataset = dataset.map(format_func, remove_columns=original_columns)

    print(f"Esempio Chosen: {dataset['train'][0]['chosen'][-100:]}")
    print(f"Esempio Rejected: {dataset['train'][0]['rejected'][-100:]}")

    # 4. Modello Reward
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=1,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Fix compatibilità DistilRoBERTa
    original_forward = model.forward
    def forward_wrapper(*args, **kwargs):
        kwargs.pop("use_cache", None) 
        return original_forward(*args, **kwargs)
    model.forward = forward_wrapper
    model.config.use_cache = False

    # 5. Configurazione Training
    training_args = RewardConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,     
        gradient_accumulation_steps=4,     
        learning_rate=5e-5,                
        num_train_epochs=3,                
        fp16=True,                         
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="epoch",
        max_length=512, 
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=False
    )

    # 6. Trainer
    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer, 
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    print("🚀 Avvio Training Reward Model...")
    trainer.train()
    
    print(f"Salvataggio Reward Model in {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Reward Model completato! Ora giudica il CONTENUTO, non i tag.")

if __name__ == "__main__":
    train_reward_model()