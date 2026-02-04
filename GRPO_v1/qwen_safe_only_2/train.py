import os
import re
import torch
import sys
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers.cache_utils import DynamicCache

def apply_cache_patches():
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: getattr(self, "_max_cache_length", 4096)
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, seq_len=None, idx=0: self.get_seq_length(idx)

apply_cache_patches()

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

# --- REWARD FUNCTIONS ---

def extract_content(text):
    match = re.search(r"<safe_rule>(.*?)</safe_rule>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).lower().strip() if match else ""

def reward_xml_format_fn(completions, **kwargs):
    rewards = []
    for content in completions:
        if "<safe_rule>" in content and "</safe_rule>" in content:
            if len(extract_content(content)) > 5:
                rewards.append(1.0)
            else:
                rewards.append(-1.0) 
        else:
            rewards.append(-3.0)
    return rewards

# --- MODIFICA QUI: ARGOMENTO 'target' INVECE DI 'completion' ---
def reward_similarity_to_ground_truth_fn(completions, target, **kwargs):
    """
    Args:
        completions: Risposte generate dal modello.
        target: La risposta 'Ground Truth' (ora chiamata 'target' nel dataset).
    """
    rewards = []
    stop_words = {"the", "a", "an", "if", "then", "of", "to", "is", "and", "or", "for", "on", "in", "with", "by", "fail", "rule"}

    # Usiamo 'target' invece di 'completion'
    target_texts = [extract_content(c) for c in target] 

    for gen_text_raw, target_text in zip(completions, target_texts):
        gen_text = extract_content(gen_text_raw)
        score = 0.0
        
        if not gen_text:
            rewards.append(-5.0)
            continue
            
        gen_words = set(re.findall(r'\w+', gen_text)) - stop_words
        target_words = set(re.findall(r'\w+', target_text)) - stop_words
        
        intersection = gen_words.intersection(target_words)
        
        if len(target_words) > 0:
            overlap_ratio = len(intersection) / len(target_words)
            score += (overlap_ratio * 4.0) 
        
        extra_words = len(gen_words - target_words)
        if extra_words > 10: 
            score -= 1.0 

        rewards.append(score)
        
    return rewards
# ---------------------------------------------------------------

def reward_technical_quality_fn(completions, original_trigger, original_action, **kwargs):
    rewards = []
    mitigation_verbs = ["limit", "restrict", "ensure", "disable", "enable", "verify", "notify", "ask", "schedule", "encrypt", "avoid", "require", "set"]
    banned_phrases = ["check conditions", "check rules", "if safe", "make sure", "be careful"]
    
    for content, trig, act in zip(completions, original_trigger, original_action):
        score = 0.0
        safe_text = extract_content(content)
        
        if safe_text:
            if any(phrase in safe_text for phrase in banned_phrases):
                score -= 4.0
            
            context_words = set(re.findall(r'\w+', (trig + " " + act).lower()))
            safe_words = set(re.findall(r'\w+', safe_text))
            if len(safe_words.intersection(context_words)) >= 1:
                score += 1.0
            else:
                score -= 2.0 

            if any(verb in safe_text for verb in mitigation_verbs):
                score += 1.0     
        else:
            score -= 2.0
        rewards.append(score)
    return rewards

# --- CONFIG TRAINING ---

model_id = "Qwen/Qwen2.5-3B-Instruct"
dataset_path = "../dataset_qwen_safe_only_v2" # <--- NUOVA CARTELLA

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    trust_remote_code=True
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

training_args = GRPOConfig(
    output_dir="./qwen_safe_specialist_results",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8, 
    max_prompt_length=768,       
    max_completion_length=200,   
    num_generations=4,
    temperature=0.9,
    max_steps=350,               
    logging_steps=5,
    save_steps=100,
    report_to="none",
    bf16=True
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[
        reward_xml_format_fn, 
        reward_similarity_to_ground_truth_fn, 
        reward_technical_quality_fn           
    ], 
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
    print("🚀 Lancio Training Qwen 3B - Mode: Ground Truth Comparison")
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()
    trainer.train()
    trainer.save_model("./model_qwen_safe_final")
    tokenizer.save_pretrained("./model_qwen_safe_final")