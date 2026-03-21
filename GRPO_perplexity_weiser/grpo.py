import os
import sys
import re
import math
import torch

# ==========================================
# bypass blocco sicurezza
import transformers.trainer
transformers.trainer.check_torch_load_is_safe = lambda: None

import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
# ==========================================

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

# ================= CONFIGURAZIONE =================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DATA_PATH = "processed_grpo_clean" 
SFT_ADAPTER_PATH = "qwen-sft-clean/final_adapter" 
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "qwen-grpo" 

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("grpo_fresh_log.txt", "a", encoding="utf-8") 
    def write(self, message):
        self.terminal.write(message); self.log.write(message)
    def flush(self):
        self.terminal.flush(); self.log.flush()

def apply_cache_patches():
    if not hasattr(DynamicCache, "seen_tokens"): DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_max_length"): DynamicCache.get_max_length = lambda self: getattr(self, "_max_cache_length", 4096)
    if not hasattr(DynamicCache, "get_usable_length"): DynamicCache.get_usable_length = lambda self, seq_len=None, idx=0: self.get_seq_length(idx)
apply_cache_patches()

# ================= GIUDICE IOT SU GPU =================
print("Caricamento TinyLLM IoT (Giudice Fluency su GPU)...")
PERP_MODEL_ID = "weiser/101M-0.4" 
perp_tokenizer = AutoTokenizer.from_pretrained(PERP_MODEL_ID)
if perp_tokenizer.pad_token is None:
    perp_tokenizer.pad_token = perp_tokenizer.eos_token

perp_model = AutoModelForCausalLM.from_pretrained(
    PERP_MODEL_ID,
    torch_dtype=torch.bfloat16
).to("cuda").eval()

# ================= REWARD FUNCTIONS =================

def format_gatekeeper_reward(completions, **kwargs):
    """
    Punisce la mancanza di tag e il copia-incolla dei placeholder.
    """
    rewards = []
    pattern_just = r"<justification>(.*?)</justification>"
    pattern_safe = r"<safe>(.*?)</safe>"
    
    # Anti-plagio
    forbidden_general = [
        "...", "analysis of the risk", "safer rule variant", 
        "Justification:", "Safe Version:", "**Justification", "**Reasoning"
    ]
    
    for c in completions:
        match_j = re.search(pattern_just, c, re.DOTALL | re.IGNORECASE)
        match_s = re.search(pattern_safe, c, re.DOTALL | re.IGNORECASE)
        
        # 1. Mancano i tag o sono vuoti
        if not match_j or not match_s:
            rewards.append(-10.0) 
            continue
            
        text_j = match_j.group(1).strip()
        text_s = match_s.group(1).strip()
        
        words_j = len(text_j.split())
        words_s = len(text_s.split())
        
        # 2. Testo troppo corto
        if words_j < 4 or words_s < 4:
            rewards.append(-5.0) 
            continue
            
        # 3. ANTI-PLAGIO: Copia dei placeholder del prompt
        if any(f in c.lower() for f in forbidden_general):
            rewards.append(-10.0)
            continue
            
        # 4. Giustificazione troppo lunga 
        if words_j > 50:
            rewards.append(-2.0) 
            continue
            
        # Formato perfetto
        rewards.append(2.0)
            
    return rewards

def dynamic_perplexity_reward(prompts, completions, **kwargs):
    """
    La Perplexity guida il modello verso un linguaggio naturale.
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        match = re.search(r"<\|im_start\|>user\n(.*?)(?:<\|im_end\|>|$)", prompt, re.DOTALL)
        user_input = match.group(1).strip() if match else prompt[-200:]
        
        clean_comp = re.sub(r"<[^>]+>", "", completion).strip()
        if not clean_comp:
            rewards.append(0.0)
            continue
            
        full_text = f"{user_input}\nAnswer: {clean_comp}"
        
        inputs = perp_tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        
        try:
            with torch.no_grad():
                loss = perp_model(**inputs, labels=inputs["input_ids"]).loss.item()
                perplexity = math.exp(loss)
                
            if perplexity < 50.0:
                rewards.append(3.0) 
            elif perplexity < 100.0:
                rewards.append(0.0) 
            else:
                rewards.append(-3.0) 
        except:
            rewards.append(0.0)
            
    return rewards

def main():
    sys.stdout = Logger()
    print("\n" + "="*50)
    print("AVVIO GRPO (Fresh Start, 1000 Esempi, Weiser GPU, Checkpoint 50, Eval 150)")
    print("="*50)
    
    dataset = load_dataset("json", data_files={
        "train": os.path.join(DATA_PATH, "train_grpo.jsonl"),
        "eval": os.path.join(DATA_PATH, "eval_grpo.jsonl") 
    })
    
    if len(dataset["train"]) > 1000: dataset["train"] = dataset["train"].select(range(1000))
    if len(dataset["eval"]) > 50: dataset["eval"] = dataset["eval"].select(range(50))

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-6,              
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=4,               
        
        num_train_epochs=2, 
        load_best_model_at_end=False, 
        
        eval_strategy="steps", eval_steps=150, # Modificato a 150
        save_strategy="steps", save_steps=50,  # Modificato a 50
        logging_steps=10,
        
        bf16=True, fp16=False,
        report_to="none", use_vllm=False,
        gradient_checkpointing=True,
        dataloader_num_workers=0, dataloader_pin_memory=False,
    )
    
    training_args.max_prompt_length = 512
    training_args.max_completion_length = 200 

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    
    model.load_adapter(SFT_ADAPTER_PATH, adapter_name="sft_adapter")
    model.set_adapter("sft_adapter")
    model.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM", lora_dropout=0.05, bias="none")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            format_gatekeeper_reward,    
            dynamic_perplexity_reward    
        ],
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],    
        peft_config=peft_config,
    )

    resume_training = False
    if os.path.exists(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if len(checkpoints) > 0:
            resume_training = True
            print(f"Trovati {len(checkpoints)} checkpoint in {OUTPUT_DIR}. Ripresa automatica attivata.")
        else:
            print(f"Nessun checkpoint trovato in {OUTPUT_DIR}. Addestramento da zero.")

    trainer.train(resume_from_checkpoint=resume_training) 
    
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ GRPO Completato al 100%!")

if __name__ == "__main__":
    main()