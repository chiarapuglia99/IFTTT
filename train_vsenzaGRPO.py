import os
import torch
import logging
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType

# --- CONFIGURAZIONE FILE DI LOG ---
LOG_FILE = "training_log.txt"

# Pulizia file log precedente (se esiste, lo cancella per partire puliti)
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

# Setup Logger di base (per i messaggi di sistema tipo "Caricamento modello")
logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- QUESTA È LA PARTE MODIFICATA PER I TUOI LOG ---
class FileLoggingCallback(TrainerCallback):
    """
    Questa classe intercetta ogni volta che il modello stampa una loss
    e la scrive fisicamente nel file di testo così com'è.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # 1. Stampa a console (così vedi che succede)
            print(f"LOG CATTURATO: {logs}")
            
            # 2. Scrittura SU FILE (senza orari o 'INFO', solo i dati)
            try:
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    # Scriviamo il dizionario convertito in stringa
                    f.write(str(logs) + "\n")
            except Exception as e:
                print(f"Errore scrittura log: {e}")

# --- CONFIGURAZIONE ---
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_PATH = "dataset_final_processed"
OUTPUT_DIR = "./output_llama_standard"

print(f"I log saranno salvati in: {LOG_FILE}")

# 2. CARICAMENTO DATASET
logger.info("Caricamento Dataset...")
try:
    dataset = load_from_disk(DATASET_PATH)
    logger.info(f"Dataset caricato: Train={len(dataset['train'])}")
except Exception as e:
    print(f"ERRORE: {e}")
    exit()

# 3. CARICAMENTO MODELLO
logger.info("Caricamento Modello...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16 
)

model.gradient_checkpointing_enable() 
model.enable_input_require_grads()

# 4. LORA
logger.info("Applicazione LoRA...")
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 5. ARGOMENTI DI TRAINING
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=4, 
    
    # --- PARAMETRI DI TRAINING ---
    num_train_epochs=3,            
    learning_rate=1e-4,            
    warmup_ratio=0.03,             
    weight_decay=0.01,             
    # -----------------------------
    
    fp16=True,                    
    logging_steps=10,              
    save_strategy="epoch",
    eval_strategy="steps",   
    eval_steps=50,                 
    report_to="none",
    save_total_limit=1,            
    dataloader_pin_memory=False    
)

# 6. TRAINER
trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"], 
    args=args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    
    # QUI colleghiamo la classe che scrive i log
    callbacks=[FileLoggingCallback()] 
)

logger.info("Avvio Training...")
trainer.train()

logger.info("Salvataggio modello...")
model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
print("✅ TUTTO OK! Controlla training_log.txt")