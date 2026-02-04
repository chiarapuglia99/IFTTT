import os
import re
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainerCallback
)
from transformers.cache_utils import DynamicCache
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from rouge_score import rouge_scorer

# --- MONKEY PATCH PER STABILITÀ ---
# Queste patch risolvono incompatibilità specifiche tra DynamicCache e Llama 3.2
def apply_monkey_patches():
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: getattr(self, "_max_cache_length", 4096)
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, *args, **kwargs: self.get_seq_length()
    if not hasattr(DynamicCache, "get_max_cache_shape"):
        DynamicCache.get_max_cache_shape = lambda self: getattr(self, "_max_cache_length", 4096)

apply_monkey_patches()

# 1. CONFIGURAZIONE COSTANTI E MODELLO
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
ALLOWED_CATEGORIES = ["personal harm", "physical harm", "cybersecurity harm"]
ALLOWED_TAGS = ["category", "justification", "safe_rule"]
DATASET_PATH = "../dataset_llama" # Percorso del dataset preparato con Few-Shot

print(f"Caricamento modello {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map={"": 0}, # Carica sulla prima GPU disponibile
)
model.config.use_cache = False # Disabilitato per il training (risparmio memoria)

# 2. FUNZIONI DI REWARD (GRPO)
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Helper per estrarre il contenuto tra tag XML
def extract_xml(text, tag):
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None

# Reward per la correttezza del formato XML
def reward_strict_format_fn(completions, **kwargs):
    rewards = []
    for content in completions:
        score = 0.0
        # Penalità se vengono inventati tag non autorizzati
        all_found = re.findall(r"<(/?\w+)>", content)
        found_names = set(t.replace("/", "") for t in all_found)
        for tag in found_names:
            if tag not in ALLOWED_TAGS: score -= 3.0
        
        # Controllo presenza e lunghezza minima del contenuto nei tag obbligatori
        for t in ALLOWED_TAGS:
            inner = extract_xml(content, t)
            if inner:
                if len(inner) < 4 or "..." in inner: score -= 2.0
                else: score += 1.5
            else: score -= 1.5 # Penalità se il tag manca
        rewards.append(score)
    return rewards

# Reward per l'accuratezza della categoria e anti-collasso
def reward_accuracy_strict_fn(completions, target_category, **kwargs):
    rewards = []
    # Analisi della diversità nel batch (evita che il modello risponda sempre la stessa cosa)
    batch_preds = [extract_xml(c, "category").lower().strip() if extract_xml(c, "category") else "none" for c in completions]
    
    for content, target in zip(completions, target_category):
        pred = extract_xml(content, "category")
        target_clean = target.lower().strip()
        score = 0.0
        if pred:
            pred_clean = pred.lower().strip()
            if pred_clean == target_clean: score += 12.0 # Premio alto per categoria corretta
            elif pred_clean in ALLOWED_CATEGORIES: score += 1.0 # Piccolo premio se valida ma errata
            else: score -= 6.0 # Penalità per categorie inventate
        else:
            score -= 4.0
        
        # Penalità se tutte e 4 le risposte generate per lo stesso prompt sono uguali
        if batch_preds.count(batch_preds[0]) == len(batch_preds):
            score -= 3.0
        rewards.append(score)
    return rewards

# Reward per forzare la diversità testuale tra le generazioni
def reward_diversity_fn(completions, **kwargs):
    rewards = []
    for i in range(len(completions)):
        content = completions[i]
        # Penalizza i duplicati esatti nel batch di generazione
        duplicates = sum(1 for j, other in enumerate(completions) if i != j and content == other)
        rewards.append(-4.0 * duplicates)
    return rewards

# Reward semantica per lo stile della giustificazione e qualità della safe_rule
def reward_semantic_fn(completions, target_safe, **kwargs):
    rewards = []
    for content, target in zip(completions, target_safe):
        just = extract_xml(content, "justification")
        score = 0.0
        # Bonus se la giustificazione segue il template linguistico richiesto
        if just and just.lower().startswith("this rule might cause a"):
            score += 3.0
        
        # Valutazione ROUGE-L sulla qualità della safe_rule proposta
        safe_gen = extract_xml(content, "safe_rule")
        if safe_gen and safe_gen != "..." and len(safe_gen) > 10:
            r_score = scorer.score(target, safe_gen)['rougeL'].fmeasure
            score += (r_score * 5.0)
        else:
            score -= 1.0
        rewards.append(score)
    return rewards

# 3. CALLBACK PER IL LOGGING (Salvataggio Loss e Reward)
class SaveLogsCallback(TrainerCallback):
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(f"Step: {state.global_step} - {logs}\n")
            if "loss" in logs:
                print(f"Step {state.global_step} | Loss: {logs['loss']:.4f} | Reward: {logs.get('reward', 0):.2f}")

# 4. CONFIGURAZIONE GRPO
dataset = load_from_disk(DATASET_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

training_args = GRPOConfig(
    output_dir="./output_llama_final",
    learning_rate=5e-5,             # Passo di apprendimento
    per_device_train_batch_size=1,  # Dimensione batch per GPU
    gradient_accumulation_steps=4,  # Accumulo gradienti per batch virtuale più grande
    num_generations=4,              # Numero di risposte generate per ogni prompt per calcolare il vantaggio
    max_prompt_length=1024,         # Lunghezza max input (necessaria per Few-Shot)
    max_completion_length=250,      # Lunghezza max output
    temperature=1.0,                # Creatività per esplorare diverse soluzioni
    beta=0.1,                       # Controllo della divergenza dal modello originale (KL)
    fp16=True,                      # Precisione mista (16-bit) per risparmio memoria
    logging_steps=1,
    max_steps=350,                  # Numero totale di step di training
    report_to="none"
)

# Inizializzazione del Trainer GRPO
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[
        reward_strict_format_fn, 
        reward_accuracy_strict_fn, 
        reward_diversity_fn,
        reward_semantic_fn
    ],
    args=training_args,
    train_dataset=dataset["train"],
    peft_config=LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
        task_type="CAUSAL_LM"
    ),
    callbacks=[SaveLogsCallback("log_training.txt")]
)

# 5. ESECUZIONE TRAINING
if __name__ == "__main__":
    try:
        torch.cuda.empty_cache() # Pulisce la memoria GPU prima di iniziare
        print(f"Lancio Training su {DATASET_PATH}...")
        trainer.train()
        trainer.save_model("./model_final_llama") # Salva l'adapter finale
        print("\nTraining completato con successo!")
    except Exception as e:
        print(f"\nErrore: {e}")