import os
import torch
import sys
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model
from transformers.cache_utils import DynamicCache

# --- 1. PATCH DI STABILITÀ ---
def apply_cache_patches():
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: getattr(self, "_max_cache_length", 4096)
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, seq_len=None, idx=0: self.get_seq_length(idx)

apply_cache_patches()

# --- 2. LOGGER ---
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("training_log_grpo.txt", "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()

# --- CONFIGURAZIONE PERCORSI ---
# Usiamo i modelli che abbiamo appena addestrato negli step 2 e 3
MODEL_ID = "sft_xml_output"          # Il modello che sa usare l'XML
REWARD_MODEL_ID = "reward_model_output" # Il Giudice che valuta il contenuto
DATA_DIR = "dataset_modello_grpo"
OUTPUT_DIR = "grpo_final_output"

# --- 3. CARICAMENTO REWARD MODEL (Il Giudice) ---
print("Caricamento Reward Model (Judge)...")
try:
    judge_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_ID)
    judge_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_ID,
        device_map="auto", 
        torch_dtype=torch.float16
    )
    judge_model.eval()
except OSError:
    print(f"ERRORE: Non trovo {REWARD_MODEL_ID}. Hai eseguito lo step 03?")
    sys.exit(1)

# --- 4. DEFINIZIONE REWARD FUNCTIONS ---

def safety_judge_reward(prompts, completions, **kwargs):
    """
    Usa il Reward Model per giudicare la sicurezza del contenuto.
    Il RM è stato addestrato su {Prompt + \n + Response}, quindi ricostruiamo quel formato.
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # Ricostruiamo l'input esattamente come visto dal Reward Model
        text = f"{prompt}\n{completion}"
        
        inputs = judge_tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(judge_model.device)
        
        with torch.no_grad():
            outputs = judge_model(**inputs)
            # Logits alto = Safe (Chosen), Basso = Risky (Rejected)
            score = outputs.logits[0].item()
        rewards.append(score)
    return rewards

def xml_structure_reward(completions, **kwargs):
    """
    Verifica che il modello rispetti il formato XML appreso nell'SFT.
    """
    rewards = []
    for completion in completions:
        score = 0.0
        # Deve aprire il tag
        if "<safe_rule>" in completion:
            score += 0.5
        else:
            score -= 1.0 # Penalità grave (ha dimenticato l'SFT)
            
        # Deve chiudere il tag (Segnale che la frase è finita)
        if "</safe_rule>" in completion:
            score += 1.0
        else:
            score -= 1.0 
            
        rewards.append(score)
    return rewards

def brevity_reward(completions, **kwargs):
    """
    Premia la concisione. Una regola di sicurezza deve essere diretta.
    """
    rewards = []
    for completion in completions:
        # Puliamo i tag per contare le parole vere
        clean_text = completion.replace("<safe_rule>", "").replace("</safe_rule>", "").strip()
        word_count = len(clean_text.split())
        
        score = 0.0
        # Target: 10-30 parole
        if 10 <= word_count <= 35:
            score += 0.5
        elif word_count > 40:
            # Penalità leggera se si dilunga troppo
            score -= (word_count - 40) * 0.05
            
        rewards.append(score)
    return rewards

# --- 5. TRAINING FUNCTION ---
def train_grpo():
    print(f"--- Inizio GRPO Training (Final Version) ---")

    # 1. Caricamento Dataset
    data_files = {"train": os.path.join(DATA_DIR, "train.jsonl")}
    dataset = load_dataset("json", data_files=data_files)

    # LITE MODE per velocità
    print("⚠️  LITE MODE: Riduzione dataset a 400 esempi...")
    if len(dataset["train"]) > 400:
        dataset["train"] = dataset["train"].select(range(400))

    # 2. Configurazione Modello SFT
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, 
    )

    print(f"Caricamento Modello SFT da {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )
    
    model.gradient_checkpointing_enable() 
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. LoRA Config per GRPO
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )
    model = get_peft_model(model, peft_config)

    # 4. Configurazione GRPO
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-6,           # LR basso e stabile
        adam_beta1=0.9,
        adam_beta2=0.99,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=8, 
        num_generations=4,            # 4 varianti per prompt
        
        # --- PARAMETRI CRUCIALI ---
        # Prompt Lunghi (contengono la chat history) -> 512-768
        max_prompt_length=768,      
        # Output Brevi (la regola safe) -> 200 bastano e avanzano
        max_completion_length=200, 
        # Temperature alta per esplorazione
        temperature=1.0,           
        # --------------------------

        max_steps=300,                
        save_steps=50,                
        logging_steps=5,
        bf16=True,
        report_to="none",
        use_vllm=False,
    )

    # 5. Trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        # Ordine Reward: Struttura XML -> Sicurezza (Giudice) -> Brevità
        reward_funcs=[xml_structure_reward, safety_judge_reward, brevity_reward],
        args=training_args,
        train_dataset=dataset["train"],
    )

    print("🚀 Avvio Training GRPO...")
    trainer.train()
    
    print(f"Salvataggio Modello Finale in {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("GRPO Training completato! Missione Compiuta.")

if __name__ == "__main__":
    train_grpo()