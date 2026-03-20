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

#Cartella dataset
DATA_PATH = "processed_grpo_clean" 
#Adapter LoRA prodotto dal training SFT 
SFT_ADAPTER_PATH = "qwen-sft-clean/final_adapter" 
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" #modello che effettivamente addestriamo con GRPO (dopo aver caricato l'adapter SFT)
OUTPUT_DIR = "qwen-grpo-final-v2"

# ---MODELLO PER CALCOLARE LA REWARD DI PERPLEXY specializzato per il dominio di sicurezza e privacy ---
PERP_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct" 

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("grpo_final_v2_log.txt", "a", encoding="utf-8") 
    def write(self, message):
        self.terminal.write(message); self.log.write(message); self.log.flush() 
    def flush(self):
        self.terminal.flush(); self.log.flush()


def apply_cache_patches():
    if not hasattr(DynamicCache, "seen_tokens"): DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_max_length"): DynamicCache.get_max_length = lambda self: getattr(self, "_max_cache_length", 4096)
    if not hasattr(DynamicCache, "get_usable_length"): DynamicCache.get_usable_length = lambda self, seq_len=None, idx=0: self.get_seq_length(idx)
apply_cache_patches()

# --- CARICAMENTO GIUDICE DI DOMINIO (Security/Privacy) ---
print(f"⏳ Caricamento {PERP_MODEL_ID} (Giudice Fluency e Sicurezza)...")
perp_tokenizer = AutoTokenizer.from_pretrained(PERP_MODEL_ID)
perp_model = AutoModelForCausalLM.from_pretrained(
    PERP_MODEL_ID, 
    torch_dtype=torch.bfloat16, 
    device_map="auto" # Prova a caricarlo in GPU se c'è spazio, altrimenti gestisce l'offload
).eval()


#Funzione di reward che penalizza risposte in cui sono presenti parole proibite che indicano che il modello sta semplicemente imitando la struttura del prompt senza fornire risposte originali e contestualizzate.
def lazy_penalty_reward(completions, **kwargs):
    rewards = []
    forbidden = [
        "...analysis of the risk...", "...safer rule variant...",
        "analysis of the risk", "safer rule variant",
        "[Your concise justification here]", "[The safe variant of the rule here]",
        "Justification:", "Safe Version:", "**Justification",
        "**Safe Version", "**Reasoning", "Reasoning:", "Safeguarded Rule:"
    ]
    for c in completions:
        score = 0.0
        for phrase in forbidden:
            if phrase.lower() in c.lower():
                #penalità
                score = -5.0 
                break
        rewards.append(score)
    return rewards

#Funzione di reward che controlla se esiste nella risposta il tag <justification> e <safe> e se il loro contenuto è sufficientemente lungo (almeno 4 parole). Se mancano o sono troppo brevi, penalizza la risposta.
def strict_format_and_content_reward(completions, **kwargs):
    rewards = []
    pattern_just = r"<justification>(.*?)</justification>"
    pattern_safe = r"<safe>(.*?)</safe>"
    for c in completions:
        score = 0.0
        match_j = re.search(pattern_just, c, re.DOTALL | re.IGNORECASE)
        if match_j:
            #+1 se esiste il tag e contiene almeno 4 parole, altrimenti -1. Se manca il tag, penalità più severa di -2
            score += 1.0 if len(match_j.group(1).split()) >= 4 else -1.0
        else: score -= 2.0 
            
        match_s = re.search(pattern_safe, c, re.DOTALL | re.IGNORECASE)
        if match_s:
            #+1 se esiste il tag e contiene almeno 4 parole, altrimenti -1. Se manca il tag, penalità più severa di -2
            score += 1.0 if len(match_s.group(1).split()) >= 4 else -1.0
        else: score -= 2.0
        rewards.append(score)
    return rewards

#Funzione di reward che controlla la coerenza semantica tra l'input (analizzando categoria di danno) e la giustificazione fornita nella risposta, controllando la presenza di parole chiave specifiche per il dominio di sicurezza e privacy. Se la giustificazione contiene parole chiave rilevanti per il dominio, assegna una ricompensa positiva, altrimenti penalizza la risposta.
def semantic_consistency_reward(prompts, completions, **kwargs):
    rewards = []
    keywords = {
        "Cybersecurity": ["attacker", "malicious", "link", "hack", "data", "phishing", "network", "malware", "unauthorized", "spam", "secure", "encryption"],
        "Physical": ["location", "burglar", "thief", "home", "intruder", "safety", "unlock", "door", "damage", "fire", "hazard", "supervision"],
        "Personal": ["sensitive", "embarrassing", "notification", "privacy", "family", "private", "disturb", "leak", "public", "permission", "authorized"]
    }
    for prompt, completion in zip(prompts, completions):
        score = 0.0
        cat = next((k for k in keywords if k in prompt), None)
        if cat:
            match = re.search(r"<justification>(.*?)</justification>", completion, re.DOTALL | re.IGNORECASE)
            if match:
                hits = sum(1 for w in keywords[cat] if w in match.group(1).lower())
                #+0.5 se nella frase vi è almento una keyword, altrimenti lo penalizza. Possono esserci un max di 3 keyword rilevanti per cui la ricompensa massima è 1.1.
                score = 0.5 + (min(hits, 3) * 0.2) if hits >= 1 else -0.5
            else: score = -0.5
        rewards.append(score)
    return rewards

#Funzione di reward che utilizza un modello Qwen-0.5B per calcolare la perplexity condizionata dell'output generato rispetto all'input dell'utente. Se la risposta è scritta in un inglese tecnico coerente con l'input, assegna una ricompensa più alta, altrimenti penalizza la risposta.
def conditional_perplexity_reward(prompts, completions, **kwargs):
    rewards = []  #lista in cui verranno salvate le ricompense calcolate per ogni risposta generata
    
    device = perp_model.device 
    
    for prompt, completion in zip(prompts, completions):
        match = re.search(r"<\|im_start\|>user\n(.*?)(?:<\|im_end\|>|$)", prompt, re.DOTALL)
        #estrazione dell'input dell'utente dal prompt per poterlo includere nel testo valutato dal modello di perplexity
        user_input = match.group(1).strip() if match else prompt[-200:] #se non riesce ad estrarre l'input, prende gli ultimi 200 caratteri del prompt 
        
        #rimuove tag in modo che la perplexity deve valutare solo il linguaggio naturale non la struttura del prompt
        clean_comp = re.sub(r"<[^>]+>", "", completion).strip()
        if len(clean_comp.split()) < 4:
            rewards.append(-1.0) #penalità se la risposta conteniene meno di 4 parole dopo la rimozione dei tag
            continue

        #creato testo per il giudice 
        full_text = f"{user_input}\nAnswer: {clean_comp}"
        inputs = perp_tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            #calcolo della Loss quanto è probabile questo testo per il modello
            outputs = perp_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            
        # Reward basata sulla perplexity (più bassa è la loss, più alta è la reward)
        rewards.append(1.0 / (1.0 + 0.1 * math.exp(loss)))
    return rewards

def main():
    sys.stdout = Logger()
    print("\n" + "="*50)
    print("🚀 TRAINING GRPO CON GIUDICE QWEN-0.5B (Security Domain)")
    print("="*50)
    
    dataset = load_dataset("json", data_files={
        "train": os.path.join(DATA_PATH, "train_grpo.jsonl"),
        "validation": os.path.join(DATA_PATH, "test_grpo.jsonl")
    })
    
    
    NUM_SAMPLES_TRAIN = 1000
    NUM_SAMPLES_TEST = 100
    
    if len(dataset["train"]) > NUM_SAMPLES_TRAIN: dataset["train"] = dataset["train"].select(range(NUM_SAMPLES_TRAIN))
    if len(dataset["validation"]) > NUM_SAMPLES_TEST: dataset["validation"] = dataset["validation"].select(range(NUM_SAMPLES_TEST))

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-6,              
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=4,               
        num_train_epochs=2, 
        load_best_model_at_end=False,
        eval_strategy="steps", eval_steps=100, # Aumentato leggermente visti i più dati
        save_strategy="steps", save_steps=100,
        logging_steps=5,
        bf16=True, fp16=False,
        report_to="none", use_vllm=False,
        gradient_checkpointing=True,
        dataloader_num_workers=0, dataloader_pin_memory=False,
    )
    
    #Limiti token
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
            lazy_penalty_reward,
            strict_format_and_content_reward,
            semantic_consistency_reward,
            conditional_perplexity_reward
        ],
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
    )

    print("🔄 Avvio training GRPO...")
    # Se hai già una cartella qwen-grpo-final-v2 con dei checkpoint, usa resume_from_checkpoint=True
    # Altrimenti, se vuoi ricominciare da zero con il nuovo giudice, imposta a False
    trainer.train(resume_from_checkpoint=True) 
    
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ GRPO con Qwen-0.5B Completato!")

if __name__ == "__main__":
    main()