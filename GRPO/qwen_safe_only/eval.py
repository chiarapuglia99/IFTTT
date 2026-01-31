import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import re
import pandas as pd

# --- CONFIGURAZIONE ---
BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "./model_qwen_safe_final" # La cartella dove train_safe_only.py ha salvato il modello
DATASET_PATH = "../dataset_qwen_safe_only" 
OUTPUT_REPORT = "report_safe_specialist.txt"
OUTPUT_CSV = "results_safe_specialist.csv"

# Numero di campioni da testare (None = tutti)
NUM_SAMPLES = None 

# Verbi che indicano una mitigazione attiva (Lo stesso elenco usato nel training)
MITIGATION_VERBS = [
    "limit", "restrict", "ensure", "disable", "enable", 
    "verify", "notify", "ask", "schedule", "encrypt", 
    "avoid", "check", "monitor", "require", "set", "manual"
]

def extract_safe_rule(text):
    """Estrae il contenuto tra i tag <safe_rule>"""
    match = re.search(r"<safe_rule>(.*?)</safe_rule>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None

def evaluate_quality(safe_rule, trigger, action):
    """
    Assegna un punteggio di qualità alla regola generata.
    Ritorna un dizionario con i dettagli.
    """
    if not safe_rule:
        return {"valid_format": False, "relevant": False, "mitigating": False, "score": 0}

    safe_lower = safe_rule.lower()
    context_lower = (trigger + " " + action).lower()
    
    # 1. Check Pertinenza (Overlap parole)
    # Cerchiamo se almeno una parola significativa del contesto è presente nella regola
    context_words = set(re.findall(r'\w+', context_lower)) - {"the", "a", "an", "if", "then", "of", "to", "is"}
    safe_words = set(re.findall(r'\w+', safe_lower))
    overlap = safe_words.intersection(context_words)
    is_relevant = len(overlap) > 0

    # 2. Check Mitigazione (Verbi)
    has_mitigation = any(verb in safe_lower for verb in MITIGATION_VERBS)

    # 3. Check Lunghezza (Non troppo breve)
    is_good_length = 5 <= len(safe_words) <= 50

    # Punteggio sintetico (0-10)
    score = 0
    if is_relevant: score += 4
    if has_mitigation: score += 4
    if is_good_length: score += 2

    return {
        "valid_format": True,
        "relevant": is_relevant,
        "mitigating": has_mitigation,
        "score": score
    }

def run_evaluation():
    print("🚀 Avvio Valutazione 'Safe Rule Specialist'...")

    # 1. Caricamento Modello
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    # 2. Caricamento Dataset
    try:
        ds = load_from_disk(DATASET_PATH)["test"]
    except:
        print(f"❌ Errore: Dataset non trovato in {DATASET_PATH}")
        return

    if NUM_SAMPLES:
        ds = ds.select(range(min(NUM_SAMPLES, len(ds))))

    results = []
    print(f"🧪 Valutazione su {len(ds)} esempi...")

    for example in tqdm(ds):
        prompt = example["prompt"]
        orig_trigger = example["original_trigger"]
        orig_action = example["original_action"]
        ground_truth_safe = extract_safe_rule(example["completion"]) # Quella del dataset

        # Generazione
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1, # Basso per essere precisi
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text.split("assistant")[-1].strip()
        
        # Estrazione e Valutazione
        gen_safe_rule = extract_safe_rule(response)
        metrics = evaluate_quality(gen_safe_rule, orig_trigger, orig_action)

        results.append({
            "trigger": orig_trigger,
            "action": orig_action,
            "generated_rule": gen_safe_rule,
            "ground_truth": ground_truth_safe,
            "valid_format": metrics["valid_format"],
            "relevant": metrics["relevant"],
            "mitigating": metrics["mitigating"],
            "score": metrics["score"]
        })

    # --- CALCOLO STATISTICHE ---
    df_res = pd.DataFrame(results)
    
    total = len(df_res)
    format_ok = df_res["valid_format"].sum()
    relevant_ok = df_res["relevant"].sum()
    mitigating_ok = df_res["mitigating"].sum()
    avg_score = df_res["score"].mean()

    report = (
        f"=== REPORT VALUTAZIONE SAFE RULE SPECIALIST ===\n"
        f"Modello: {ADAPTER_PATH}\n"
        f"Campioni testati: {total}\n\n"
        f"METRICHE QUANTITATIVE:\n"
        f"1. Aderenza Formato XML:  {format_ok}/{total} ({format_ok/total:.1%})\n"
        f"2. Pertinenza Contesto:   {relevant_ok}/{total} ({relevant_ok/total:.1%}) -> La regola parla degli oggetti giusti?\n"
        f"3. Mitigazione Attiva:    {mitigating_ok}/{total} ({mitigating_ok/total:.1%}) -> Usa verbi di sicurezza (limit, ensure...)?\n"
        f"4. Punteggio Medio (0-10): {avg_score:.2f}\n\n"
        f"ESEMPI QUALITATIVI:\n"
    )

    # Aggiungiamo 10 esempi casuali al report
    examples = df_res.sample(min(10, total))
    for _, row in examples.iterrows():
        report += f"-"*50 + "\n"
        report += f"Trigger: {row['trigger']} -> Action: {row['action']}\n"
        report += f"GENERATA: {row['generated_rule']}\n"
        report += f"TARGET (Dataset): {row['ground_truth']}\n"
        report += f"Score: {row['score']}/10\n"

    print("\n" + report)
    
    # Salvataggio
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    df_res.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Report salvato in {OUTPUT_REPORT}")
    print(f"✅ Dati completi salvati in {OUTPUT_CSV}")

if __name__ == "__main__":
    run_evaluation()