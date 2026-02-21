import os
import sys
import logging
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ================= CONFIGURAZIONE =================
# MODIFICA 1: Puntiamo alla cartella dei dati PULITI
DATA_PATH = "processed_sft_clean"

# MODIFICA 2: Nuova cartella di output
OUTPUT_DIR = "qwen-sft-clean"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Log files
LOG_FILE_SYSTEM = "sft_clean_script.log"
LOG_FILE_METRICS = "sft_clean_metrics.txt"

# Parametri Training
MAX_SEQ_LENGTH = 1024
LEARNING_RATE = 2e-4
BATCH_SIZE = 2         # Basso per sicurezza su Windows
GRADIENT_ACCUMULATION = 8
NUM_EPOCHS = 3

# ================= SETUP LOGGING DI SISTEMA =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_SYSTEM, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ================= CALLBACK PER LOGGING METRICHE =================
class TextLoggerCallback(TrainerCallback):
    def __init__(self, filepath):
        self.filepath = filepath
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write("Step\tEpoch\tTrain Loss\tValidation Loss\n")
            f.write("-" * 50 + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with open(self.filepath, 'a', encoding='utf-8') as f:
                step = state.global_step
                epoch = logs.get("epoch", 0.0)
                train_loss = logs.get("loss", "")
                eval_loss = logs.get("eval_loss", "")
                
                t_loss_str = f"{train_loss:.4f}" if isinstance(train_loss, float) else ""
                e_loss_str = f"{eval_loss:.4f}" if isinstance(eval_loss, float) else ""
                
                if t_loss_str or e_loss_str:
                    f.write(f"{step}\t{epoch:.2f}\t{t_loss_str}\t\t{e_loss_str}\n")

# ================= FUNZIONE DI PRE-PROCESSING =================
def format_and_mask_dataset(examples, tokenizer):
    """
    Formatta l'input e applica la maschera -100 al prompt.
    Il prompt pulito viene letto automaticamente dai dati JSONL.
    """
    input_ids = []
    labels = []
    attention_masks = []

    for messages in examples['messages']:
        # 1. Costruzione Prompt (System + User)
        system_text = f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
        user_text = f"<|im_start|>user\n{messages[1]['content']}<|im_end|>\n"
        prompt_text = system_text + user_text
        
        # 2. Costruzione Risposta (Assistant)
        assistant_text = f"<|im_start|>assistant\n{messages[2]['content']}<|im_end|>"
        
        # 3. Testo Completo
        full_text = prompt_text + assistant_text
        
        # 4. Tokenizzazione
        tokenized_full = tokenizer(full_text, truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)
        tokenized_prompt = tokenizer(prompt_text, truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)
        
        input_id = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]
        
        # 5. Mascheratura (-100 sul prompt)
        label = input_id.copy()
        prompt_len = len(tokenized_prompt["input_ids"])
        
        if prompt_len > len(label):
            prompt_len = len(label)
            
        for i in range(prompt_len):
            label[i] = -100
            
        input_ids.append(input_id)
        labels.append(label)
        attention_masks.append(attention_mask)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_masks
    }

# ================= MAIN =================
if __name__ == "__main__":
    # Fix per allocazione memoria
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    logger.info(f"🚀 Avvio training SFT CLEAN su {MODEL_ID}")
    logger.info(f"📁 Input: {DATA_PATH} | Output: {OUTPUT_DIR}")

    # 1. Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        tokenizer.pad_token = "<|endoftext|>" 
        tokenizer.padding_side = "right"
        logger.info("✅ Tokenizer caricato.")
    except Exception as e:
        logger.error(f"❌ Errore caricamento tokenizer: {e}")
        sys.exit(1)

    # 2. Caricamento Dataset
    logger.info("⚙️  Caricamento e processing dataset...")
    try:
        raw_dataset = load_dataset("json", data_files={
            "train": os.path.join(DATA_PATH, "train.jsonl"),
            "validation": os.path.join(DATA_PATH, "validation.jsonl")
        })

        processed_dataset = raw_dataset.map(
            lambda x: format_and_mask_dataset(x, tokenizer),
            batched=True,
            remove_columns=raw_dataset["train"].column_names
        )
        logger.info("✅ Dataset processato.")
    except Exception as e:
        logger.error(f"❌ Errore processing dataset: {e}")
        sys.exit(1)

    # 3. Modello (Quantizzato con BFloat16 per stabilità su 4050)
    logger.info("🧠 Caricamento Modello...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        # MODIFICA 3: Usiamo bfloat16 per coerenza
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8
    )

    # 5. Argomenti Training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        num_train_epochs=NUM_EPOCHS,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        
        # MODIFICA 3b: BF16 True, FP16 False (Per evitare il crash dello scaler)
        bf16=True,
        fp16=False,
        
        report_to="none",
        group_by_length=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False,
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[TextLoggerCallback(LOG_FILE_METRICS)]
    )

    # 7. Avvio
    logger.info("🔥 Inizio Training SFT Clean...")
    trainer.train()

    # 8. Salvataggio
    logger.info("💾 Salvataggio modello finale...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
    logger.info("✅ Training SFT Clean completato!")