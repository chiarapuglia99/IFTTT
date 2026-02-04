import os
import re
import torch
import sys
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers.cache_utils import DynamicCache

# --- 1. PATCH DI STABILITÀ (Per evitare crash su lunghe sequenze) ---
def apply_cache_patches():
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: getattr(self, "_max_cache_length", 4096)
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, seq_len=None, idx=0: self.get_seq_length(idx)

apply_cache_patches()

# --- 2. LOGGER (Per salvare i progressi su file) ---
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("training_log_safe_specialist.txt", "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()

# --- 3. REWARD FUNCTIONS (Il cuore della tesi modificata) ---

def reward_xml_format_fn(completions, **kwargs):
    """
    Reward Strutturale: Verifica che l'output sia racchiuso nei tag corretti.
    Fondamentale per il parsing automatico.
    """
    rewards = []
    for content in completions:
        # Cerca i tag <safe_rule> e </safe_rule>
        if "<safe_rule>" in content and "</safe_rule>" in content:
            # Controlla che non sia vuoto dentro
            match = re.search(r"<safe_rule>(.*?)</safe_rule>", content, re.DOTALL)
            if match and len(match.group(1).strip()) > 5:
                rewards.append(1.0) # Formato perfetto
            else:
                rewards.append(-1.0) # Tag presenti ma contenuto vuoto
        else:
            rewards.append(-3.0) # Formato rotto
    return rewards

def reward_trigger_action_relevance_fn(completions, original_trigger, original_action, **kwargs):
    """
    Reward Semantica (Richiesta Prof): Focus su Trigger e Action.
    Premia le regole che:
    1. Sono pertinenti (usano parole del contesto originale).
    2. Sono 'attive' (usano verbi di mitigazione).
    3. Sono concise ma non troppo brevi.
    """
    rewards = []
    # Verbi forti che indicano una mitigazione attiva
    mitigation_verbs = [
        "limit", "restrict", "ensure", "disable", "enable", 
        "verify", "notify", "ask", "schedule", "encrypt", 
        "avoid", "check", "monitor", "require", "set"
    ]
    
    for content, trig, act in zip(completions, original_trigger, original_action):
        score = 0.0
        match = re.search(r"<safe_rule>(.*?)</safe_rule>", content, re.DOTALL | re.IGNORECASE)
        
        if match:
            safe_text = match.group(1).lower().strip()
            
            # --- CRITERIO A: PERTINENZA ---
            # Deve menzionare elementi del trigger o dell'action originali
            # Es: Se l'azione era "Open Door", la safe rule deve dire "Door".
            context_text = (trig + " " + act).lower()
            # Rimuoviamo caratteri speciali per confronto pulito
            safe_words = set(re.findall(r'\w+', safe_text))
            context_words = set(re.findall(r'\w+', context_text))
            
            # Intersezione (escludendo stop words comuni se necessario, ma qui semplifichiamo)
            overlap = safe_words.intersection(context_words)
            
            if len(overlap) >= 1:
                score += 1.0 # È pertinente al contesto
            else:
                score -= 2.0 # Allucinazione: parla di cose che non c'entrano

            # --- CRITERIO B: MITIGAZIONE ATTIVA ---
            # Deve contenere verbi che indicano controllo o sicurezza
            if any(verb in safe_text for verb in mitigation_verbs):
                score += 2.0 # Ottimo, sta usando un linguaggio di sicurezza
            
            # --- CRITERIO C: LUNGHEZZA OTTIMALE ---
            # Non troppo breve ("Don't do it"), non troppo lunga (poema)
            word_count = len(safe_text.split())
            if 6 <= word_count <= 40:
                score += 0.5
            elif word_count < 5:
                score -= 1.0 # Troppo breve
            
        else:
            score -= 2.0 # Manca il contenuto
            
        rewards.append(score)
    return rewards

# --- 4. CONFIGURAZIONE TRAINING ---

model_id = "Qwen/Qwen2.5-3B-Instruct"
dataset_path = "../dataset_qwen_safe_only" # Assicurati che sia la cartella corretta

# Configurazione Quantizzazione (4-bit per stare in 6GB VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Caricamento Modello Base
print(f"📥 Caricamento modello: {model_id}...")
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    trust_remote_code=True
)

model.config.use_cache = False
# Abilita checkpointing per risparmiare memoria
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Configurazione GRPO
training_args = GRPOConfig(
    output_dir="./qwen_safe_specialist_results",
    learning_rate=2e-5,          # LR leggermente più aggressivo perché il task è circoscritto
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8, 
    max_prompt_length=512,       # Spazio sufficiente per Descrizione + Diagnosi
    max_completion_length=150,   # Corto: deve generare solo la Safe Rule
    num_generations=4,           # Genera 4 varianti e impara dalla migliore
    temperature=0.9,             # Creatività per esplorare diverse soluzioni safe
    max_steps=300,               # Training breve ma intenso
    logging_steps=5,
    save_steps=100,
    report_to="none",
    bf16=True                    # Usa Bfloat16 per stabilità
)

# Inizializzazione Trainer
trainer = GRPOTrainer(
    model=model,
    # Lista delle reward functions definite sopra
    reward_funcs=[reward_xml_format_fn, reward_trigger_action_relevance_fn], 
    args=training_args,
    train_dataset=load_from_disk(dataset_path)["train"],
    peft_config=LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        task_type="CAUSAL_LM"
    ),
)

if __name__ == "__main__":
    print("🚀 Lancio Training Qwen 3B - Safe Rule Specialist Mode")
    print(f"Dataset: {dataset_path}")
    
    # Pulizia Cache CUDA
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()
    
    trainer.train()
    
    print("✅ Training completato. Salvataggio modello...")
    trainer.save_model("./model_qwen_safe_final")
    tokenizer.save_pretrained("./model_qwen_safe_final")
    print("💾 Modello salvato in ./model_qwen_safe_final")