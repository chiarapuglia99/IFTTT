import os
import torch
import sys
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers.cache_utils import DynamicCache

# --- 1. PATCH DI STABILITÀ (Anti-Crash per Qwen) ---
def apply_cache_patches():
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: getattr(self, "_max_cache_length", 4096)
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, seq_len=None, idx=0: self.get_seq_length(idx)

apply_cache_patches()

# --- 2. LOGGER (Salva output su file) ---
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("training_log_sft.txt", "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()

# --- CONFIGURAZIONE ---
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATA_DIR = "dataset_modello_grpo"
OUTPUT_DIR = "sft_xml_output"

def train_sft():
    print(f"--- Inizio SFT Training con {MODEL_NAME} ---")
    
    # 1. Caricamento Dataset
    data_files = {
        "train": os.path.join(DATA_DIR, "train.jsonl"),
        "validation": os.path.join(DATA_DIR, "val.jsonl")
    }
    
    if not os.path.exists(data_files["train"]):
        raise FileNotFoundError(f"File non trovati in {DATA_DIR}. Assicurati di aver eseguito lo step 01 aggiornato.")

    dataset = load_dataset("json", data_files=data_files)

    # 2. Configurazione Quantizzazione (4-bit per GPU 6GB)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True,
    )

    # 3. Tokenizer (Lo carichiamo PRIMA per usarlo nella formattazione)
    print("Caricamento Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "right" # Qwen preferisce right padding per training

    # --- 4. PRE-FORMATTAZIONE DATASET (ADATTATA ALLO STEP 1) ---
    print("Pre-formattazione del dataset (Concatenazione Prompt Qwen + Target XML)...")
    
    def format_row(example):
        # 'prompt' contiene già il Chat Template (<|im_start|>system...<|im_start|>assistant\n)
        prompt = example.get('prompt', '')
        # 'completion' contiene il target XML (<safe_rule>...</safe_rule>)
        comp = example.get('completion', '')
        
        # UNIONE DIRETTA: Non aggiungiamo \n perché il prompt Qwen finisce già con a capo.
        # Aggiungiamo EOS token alla fine per insegnare al modello a fermarsi.
        full_text = prompt + comp + tokenizer.eos_token
        
        return {"text": full_text}

    # Applichiamo la trasformazione
    dataset = dataset.map(format_row)
    
    # Verifica debug (stampiamo i primi caratteri e gli ultimi per vedere se c'è l'EOS)
    print(f"Esempio Prompt Inizio: {dataset['train'][0]['text'][:50]}...")
    print(f"Esempio Prompt Fine: ...{dataset['train'][0]['text'][-50:]}")

    # 5. Caricamento Modello Base
    print("Caricamento Modello...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", 
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False 

    # 6. Configurazione LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 7. Training Arguments
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,   # Batch 1 per memoria
        gradient_accumulation_steps=8,   # Accumulo gradienti
        learning_rate=2e-4,              # LR standard per Qwen LoRA
        num_train_epochs=1,              # 1 Epoca basta per imparare lo stile XML
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,                       
        optim="paged_adamw_8bit",        
        packing=False,
        report_to="none",                
        dataset_text_field="text",       
    )
    
    # Aumentato a 1024 perché abbiamo inserito i Few-Shot examples nel prompt
    training_args.max_seq_length = 1024

    # 8. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer, 
        args=training_args
    )

    print("🚀 Avvio Training SFT...")
    trainer.train()
    
    print(f"Salvataggio modello SFT in {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("SFT Training completato con successo!")

if __name__ == "__main__":
    train_sft()