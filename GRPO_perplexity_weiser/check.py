import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ================= CONFIGURAZIONE =================
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "qwen-grpo"
TEST_FILE = "processed_sft_clean/test.jsonl" # Usiamo il test set ufficiale
NUM_SAMPLES = 5

SYSTEM_PROMPT = (
    "You are an expert in IoT security and safety. "
    "Your task is to analyze an automation rule and its associated risk category. "
    "You must provide a concise justification for why the rule is considered unsafe, "
    "and generate a safer variant of the rule.\n\n"
    "Your output must strictly follow this XML format:\n"
    "<justification> ...analysis of the risk... </justification>\n"
    "<safe> ...safer rule variant... </safe>"
)

def main():
    # 1. Trova l'ultimo checkpoint automaticamente
    if not os.path.exists(OUTPUT_DIR):
        print(f"❌ Cartella {OUTPUT_DIR} non trovata.")
        return
        
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if not checkpoints:
        print("❌ Nessun checkpoint trovato in", OUTPUT_DIR)
        return
        
    # Ordina i checkpoint per numero e prende il più alto
    latest_ckpt = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    ckpt_path = os.path.join(OUTPUT_DIR, latest_ckpt)
    print(f"🔍 Trovato! Caricamento dell'ultimo checkpoint: {latest_ckpt}")

    # 2. Carica il Modello e il Checkpoint
    print("🧠 Caricamento Modello Base + LoRA Checkpoint in corso...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    
    # Applica i pesi del checkpoint appena addestrato
    model = PeftModel.from_pretrained(model, ckpt_path)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    # 3. Carica i primi 5 esempi di Test
    print(f"📄 Lettura di {NUM_SAMPLES} esempi dal test set...")
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()[:NUM_SAMPLES]

    print("\n" + "="*50)
    print("🎯 INIZIO GENERAZIONE")
    print("="*50)

    # 4. Genera le risposte
    for i, line in enumerate(lines):
        data = json.loads(line)
        user_input = data['messages'][1]['content'] # Prende la 'Description' e 'Risk Category'
        
        input_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
        
        prompt_text = tokenizer.apply_chat_template(input_msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3, 
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        print(f"\n--- ESEMPIO {i+1} ---")
        print(f"📥 INPUT:\n{user_input}")
        print(f"🤖 OUTPUT MODELLO:\n{full_output}")
        print("-" * 50)

if __name__ == "__main__":
    main()