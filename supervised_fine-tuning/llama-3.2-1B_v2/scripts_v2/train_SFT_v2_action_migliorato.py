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
    TrainerCallback,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# --- LOGGING SU FILE ---
LOG_FILE = "training_log.txt"
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

logging.basicConfig(filename=LOG_FILE, filemode='a', level=logging.INFO)
logger = logging.getLogger(__name__)

class FileLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"LOG: {logs}")
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(str(logs) + "\n")

# --- CONFIGURAZIONE ---
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_PATH = "dataset_final_processed_nobalance"
OUTPUT_DIR = "./output_llama_qlora"

# 1. CARICAMENTO DATASET
dataset = load_from_disk(DATASET_PATH)

# 2. CONFIGURAZIONE QLORA (4-bit)
# Questo permette di caricare il modello base usando poca VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Caricamento modello con QLoRA...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config, # Caricamento a 4-bit
    device_map="auto"
)

# Prepara il modello per il training (gradient checkpointing, etc.)
model = prepare_model_for_kbit_training(model)

# 3. CONFIGURAZIONE LORA
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

# 4. TRAINING ARGUMENTS
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4, # LR tipico per QLoRA
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="no",
    report_to="none"
)

# 5. TRAINER
trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    args=args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[FileLoggingCallback()]
)

print("Avvio Training...")
trainer.train()

# Salvataggio adapter
model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
print("✅ Training completato.")