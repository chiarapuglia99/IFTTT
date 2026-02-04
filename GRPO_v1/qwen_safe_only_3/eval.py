import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import re
import pandas as pd
import evaluate # Per ROUGE
from sentence_transformers import SentenceTransformer, util # Per la similarità semantica

# --- CONFIGURAZIONE ---
BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "./model_qwen_safe_final"
# MODIFICA 1: Puntiamo al dataset V2 corretto
DATASET_PATH = "../dataset_qwen_safe_only_v3" 
OUTPUT_REPORT = "report_safe_specialist_rouge.txt"
OUTPUT_CSV = "results_safe_specialist_rouge.csv"

NUM_SAMPLES = None 

# --- INIZIALIZZAZIONE METRICHE ---
print("⚙️ Caricamento metriche (ROUGE & Semantic Model)...")
rouge_metric = evaluate.load("rouge")
# Modello piccolo e veloce per calcolare se due frasi hanno lo stesso significato
semantic_model = SentenceTransformer('all-MiniLM-L6-v2') 

# Verbi di mitigazione (controllo euristico)
MITIGATION_VERBS = [
    "limit", "restrict", "ensure", "disable", "enable", 
    "verify", "notify", "ask", "schedule", "encrypt", 
    "avoid", "check", "monitor", "require", "set", "manual"
]

def extract_safe_rule(text):
    """Estrae il contenuto tra i tag <safe_rule>"""
    if not isinstance(text, str): return ""
    match = re.search(r"<safe_rule>(.*?)</safe_rule>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""

def calculate_advanced_metrics(generated, reference):
    """
    Calcola ROUGE (testuale) e Cosine Similarity (semantica).
    """
    if not generated or not reference:
        return {"rougeL": 0.0, "semantic_sim": 0.0}

    # 1. ROUGE Score (Overlap testuale)
    rouge_results = rouge_metric.compute(predictions=[generated], references=[reference])
    
    # 2. Semantic Similarity (Significato)
    # Trasforma le frasi in vettori numerici (embeddings)
    embedding_gen = semantic_model.encode(generated, convert_to_tensor=True)
    embedding_ref = semantic_model.encode(reference, convert_to_tensor=True)
    # Calcola quanto sono vicini i vettori (da 0 a 1)
    cosine_sim = util.cos_sim(embedding_gen, embedding_ref).item()

    return {
        "rougeL": rouge_results["rougeL"], # ROUGE-L è il migliore per frasi intere
        "semantic_sim": cosine_sim
    }

def run_evaluation():
    print("🚀 Avvio Valutazione 'Safe Rule Specialist' (ROUGE + Semantica)...")

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
        
        # MODIFICA 2: Usiamo 'target' invece di 'completion'
        # Questa colonna contiene la risposta dell'esperto (Ground Truth)
        ground_truth_safe = extract_safe_rule(example["target"]) 
        
        # Generazione
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text.split("assistant")[-1].strip()
        
        # Estrazione
        gen_safe_rule = extract_safe_rule(response)
        
        # Calcolo Metriche Avanzate
        adv_metrics = calculate_advanced_metrics(gen_safe_rule, ground_truth_safe)
        
        # Check Euristici (Mitigazione attiva)
        has_mitigation = any(v in gen_safe_rule.lower() for v in MITIGATION_VERBS) if gen_safe_rule else False
        valid_format = gen_safe_rule != ""

        results.append({
            "generated_rule": gen_safe_rule,
            "ground_truth": ground_truth_safe,
            "valid_format": valid_format,
            "mitigating_verb": has_mitigation,
            "rougeL": adv_metrics["rougeL"],
            "semantic_sim": adv_metrics["semantic_sim"]
        })

    # --- CALCOLO STATISTICHE FINALI ---
    df_res = pd.DataFrame(results)
    
    total = len(df_res)
    avg_rouge = df_res["rougeL"].mean()
    avg_semantic = df_res["semantic_sim"].mean()
    mitigation_pct = df_res["mitigating_verb"].mean() * 100
    format_pct = df_res["valid_format"].mean() * 100

    report = (
        f"=== REPORT AVANZATO SAFE RULE (ROUGE + SEMANTIC) ===\n"
        f"Modello: {ADAPTER_PATH}\n"
        f"Campioni testati: {total}\n\n"
        f"METRICHE DI SIMILARITÀ (Quanto imita l'esperto?):\n"
        f"1. ROUGE-L Score:      {avg_rouge:.4f}  (0=Diverso, 1=Identico parola per parola)\n"
        f"2. Semantic Similarity: {avg_semantic:.4f} (0=Diverso, 1=Stesso significato)\n"
        f"   -> Nota: Se Semantic > ROUGE, il modello usa parole diverse ma corrette.\n\n"
        f"METRICHE EURISTICHE (Qualità tecnica):\n"
        f"3. Aderenza XML:       {format_pct:.1f}%\n"
        f"4. Uso Verbi Mitigazione: {mitigation_pct:.1f}%\n\n"
        f"ESEMPI DETTAGLIATI:\n"
    )

    # Ordiniamo per similarità semantica decrescente (i migliori primi)
    examples = df_res.sample(min(15, total))
    
    for _, row in examples.iterrows():
        report += f"-"*60 + "\n"
        report += f"TARGET (Esperto): {row['ground_truth']}\n"
        report += f"GENERATA (AI):    {row['generated_rule']}\n"
        report += f"METRICHE -> ROUGE-L: {row['rougeL']:.2f} | SEMANTICA: {row['semantic_sim']:.2f}\n"
        # Piccolo commento automatico
        if row['semantic_sim'] > 0.8: report += "✅ SIGNIFICATO QUASI IDENTICO\n"
        elif row['semantic_sim'] < 0.5: report += "⚠️ SIGNIFICATO DIVERGENTE\n"

    print("\n" + report)
    
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    df_res.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Report salvato in {OUTPUT_REPORT}")
    print(f"✅ CSV salvato in {OUTPUT_CSV}")

if __name__ == "__main__":
    run_evaluation()