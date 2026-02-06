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
    BitsAndBytesConfig # <--- NECESSARIO PER QLoRA
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# --- CONFIGURAZIONE ---
LOG_FILE = "training_log.txt"
OUTPUT_DIR = "./output_llama_standard_vsenzaGRPO"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_PATH = "dataset_final_processed"

if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s - %(message)s',
    level=logging.INFO
)

class FileLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            log_str = "METRICS: " + str(logs)
            logging.info(log_str)
            print(log_str)

def main():
    dataset = load_from_disk(DATASET_PATH)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # --- CONFIGURAZIONE 4-BIT (QLoRA) PER GTX 1050 Ti ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16 # La 1050 Ti supporta float16
    )

    print("Caricamento modello in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Preparazione modello per training k-bit e checkpointing (salva RAM)
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable() 
    
    # Configurazione LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,              # Ridotto a 8 per salvare memoria
        lora_alpha=16,    # Ridotto proporzionalmente
        target_modules=["q_proj", "v_proj"], # Solo attention modules per leggerezza
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=8, # Aumentato per compensare il batch size 1
        num_train_epochs=3,            
        learning_rate=2e-4,            
        warmup_ratio=0.03,             
        weight_decay=0.01,             
        fp16=True,                    
        logging_steps=10,              
        save_strategy="epoch",
        eval_strategy="steps",   
        eval_steps=50,                 
        report_to="none",
        save_total_limit=1,
        # Ottimizzatore PAGED per gestire i picchi di memoria usando la RAM di sistema
        optim="paged_adamw_32bit" 
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"], 
        args=args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[FileLoggingCallback()]
    )
    
    print("Avvio Training ottimizzato per 4GB VRAM...")
    trainer.train()
    
    print("Salvataggio Adapter finale...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_adapter"))

if __name__ == "__main__":
    main()