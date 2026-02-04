import torch
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from bert_score import score as bert_score
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

# --- DOWNLOAD RISORSE NLTK (CORRETTO) ---
# Scarichiamo tutto il necessario per evitare errori
print("Verifica risorse NLTK...")
nltk.download('punkt')
nltk.download('punkt_tab') # <--- ECCO LA FIX
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- CONFIGURAZIONE ---
BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "grpo_final_output"     # La cartella dove hai salvato il GRPO
DATA_DIR = "dataset_modello_grpo"      # Dove sta test.jsonl

# --- 1. FUNZIONI DI UTILITÀ ---

def calculate_cli(text):
    """Calcola il Coleman-Liau Index (Readability). Più basso = più leggibile."""
    if not text: return 0.0
    text = text.strip()
    words = nltk.word_tokenize(text)
    sentences = nltk.sent_tokenize(text)
    
    num_words = len(words)
    num_sentences = len(sentences)
    num_letters = sum(len(w) for w in words if w.isalnum())
    
    if num_words == 0: return 0.0
    
    L = (num_letters / num_words) * 100
    S = (num_sentences / num_words) * 100
    
    cli = 0.0588 * L - 0.296 * S - 15.8
    return cli

def extract_xml_content(text):
    """Estrae il contenuto dentro <safe_rule>."""
    match = re.search(r"<safe_rule>(.*?)</safe_rule>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: se ha aperto ma non chiuso
    match_open = re.search(r"<safe_rule>(.*)", text, re.DOTALL | re.IGNORECASE)
    if match_open:
        return match_open.group(1).strip()
    return text.strip()

def segment_rule_smart(text):
    """
    Segmenta la regola in [Parte Principale] e [Condizione/Trigger].
    Adattato per frasi naturali IoT come: "Do X when Y, only if Z".
    """
    text_clean = text.lower().strip()
    
    # Separatori tipici del tuo dataset (in ordine di priorità)
    separators = [" but only if ", " only if ", " provided that ", " requiring ", " unless ", " when "]
    
    part_a = text_clean # Solitamente l'Action
    part_b = ""         # Solitamente il Trigger/Safety Check
    
    split_found = False
    for sep in separators:
        if sep in text_clean:
            parts = text_clean.split(sep, 1)
            part_a = parts[0].strip()
            # Reinseriamo il separatore nella seconda parte per contesto, o lo lasciamo fuori?
            # Per il confronto semantico, meglio tenerlo o toglierlo. Teniamolo pulito.
            part_b = parts[1].strip() 
            split_found = True
            break
            
    if not split_found:
        # Fallback: Se non trova separatori logici, prova a splittare sulla virgola
        if "," in text_clean:
            parts = text_clean.split(",", 1)
            part_a = parts[0].strip()
            part_b = parts[1].strip()
            
    return part_a, part_b

# --- 2. CARICAMENTO MODELLO ---
print(f"--- Inizio Valutazione ---")
print("Caricamento Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

print("Caricamento Modello Base (4-bit)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, 
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa"
)

print(f"Caricamento Adapter GRPO da {ADAPTER_PATH}...")
try:
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
except Exception as e:
    print(f"ERRORE: Impossibile caricare l'adapter. {e}")
    exit()

# --- 3. GENERAZIONE E VALUTAZIONE ---
print("Caricamento Dataset di Test...")
dataset = load_dataset("json", data_files={"test": os.path.join(DATA_DIR, "test.jsonl")})
test_data = dataset["test"]

# Liste per metriche
metrics = {
    "bert_overall": [], "bert_part_a": [], "bert_part_b": [],
    "bleu4": [], "meteor": [], "cli": []
}

results_log = []
smoothie = SmoothingFunction().method4

print(f"Valutazione su {len(test_data)} esempi...")

for i in tqdm(range(len(test_data))):
    example = test_data[i]
    
    # 1. Preparazione Input
    # Il 'prompt' nel jsonl ha già il template Qwen applicato (<|im_start|>system...)
    prompt_text = example['prompt']
    ground_truth_raw = example['completion'] # Contiene <safe_rule>...</safe_rule>
    
    # Estraiamo il contenuto pulito dal Ground Truth per i confronti
    ground_truth_clean = extract_xml_content(ground_truth_raw)
    
    # 2. Generazione
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128, # Breve, come richiesto
            do_sample=False,    # Deterministico per eval
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decodifica
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Qwen separa la risposta con <|im_start|>assistant
    if "<|im_start|>assistant" in full_output:
        prediction_raw = full_output.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
    else:
        # Fallback brutale se il template si rompe
        prediction_raw = full_output[len(prompt_text):]

    # 3. Parsing XML e Segmentazione
    prediction_clean = extract_xml_content(prediction_raw)
    
    # Segmentazione (Smart Split)
    # Part A = Main Action / Part B = Condition
    pred_a, pred_b = segment_rule_smart(prediction_clean)
    gt_a, gt_b = segment_rule_smart(ground_truth_clean)

    # --- 4. CALCOLO METRICHE ---
    
    # A. BERT Score (Semantic Similarity)
    # Overall
    try:
        _, _, b_ov = bert_score([prediction_clean], [ground_truth_clean], lang="en", verbose=False)
        metrics["bert_overall"].append(b_ov.item())
    except: metrics["bert_overall"].append(0.0)
    
    # Part A (Action/Trigger Main)
    if gt_a and pred_a:
        try:
            _, _, b_a = bert_score([pred_a], [gt_a], lang="en", verbose=False)
            metrics["bert_part_a"].append(b_a.item())
        except: metrics["bert_part_a"].append(0.0)
        
    # Part B (Safety Condition) - CRUCIALE
    if gt_b and pred_b:
        try:
            _, _, b_b = bert_score([pred_b], [gt_b], lang="en", verbose=False)
            metrics["bert_part_b"].append(b_b.item())
        except: metrics["bert_part_b"].append(0.0)
    
    # B. BLEU-4 (N-gram overlap)
    ref_tokens = nltk.word_tokenize(ground_truth_clean.lower())
    pred_tokens = nltk.word_tokenize(prediction_clean.lower())
    if len(pred_tokens) > 0:
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
        metrics["bleu4"].append(bleu)
    else:
        metrics["bleu4"].append(0.0)
        
    # C. METEOR (Synonyms)
    if len(pred_tokens) > 0:
        met = meteor_score([ref_tokens], pred_tokens)
        metrics["meteor"].append(met)
    else:
        metrics["meteor"].append(0.0)
        
    # D. CLI (Readability)
    cli_val = calculate_cli(prediction_clean)
    metrics["cli"].append(cli_val)
    
    # Logging
    results_log.append(
        f"IDX: {i}\n"
        f"GT  (XML): {ground_truth_raw}\n"
        f"PRED(XML): {prediction_raw}\n"
        f"--- CLEAN & SPLIT ---\n"
        f"GT:   [{gt_a}] | [{gt_b}]\n"
        f"PRED: [{pred_a}] | [{pred_b}]\n"
        f"{'-'*40}\n"
    )

# --- 5. REPORTING ---
avg_bert_ov = np.mean(metrics["bert_overall"])
avg_bert_a = np.mean(metrics["bert_part_a"]) if metrics["bert_part_a"] else 0.0
avg_bert_b = np.mean(metrics["bert_part_b"]) if metrics["bert_part_b"] else 0.0
avg_bleu = np.mean(metrics["bleu4"])
avg_meteor = np.mean(metrics["meteor"])
avg_cli = np.mean(metrics["cli"])

print("\n" + "="*30)
print("✅ RISULTATI FINALI")
print("="*30)
print(f"BERT-Score OVERALL Mean: {avg_bert_ov:.4f}")
print(f"BERT-Score PART A Mean : {avg_bert_a:.4f} (Main Action)")
print(f"BERT-Score PART B Mean : {avg_bert_b:.4f} (Safety Cond)")
print(f"BLEU-4 Score:            {avg_bleu:.4f}")
print(f"METEOR Score:            {avg_meteor:.4f}")
print(f"CLI (Readability) Mean:  {avg_cli:.4f}")
print("="*30)

# Salvataggio su file
with open("eval_report_final.txt", "w", encoding="utf-8") as f:
    f.write("=== METRICS REPORT ===\n")
    f.write(f"BERT-Score OVERALL: {avg_bert_ov:.4f}\n")
    f.write(f"BERT-Score PART A:  {avg_bert_a:.4f}\n")
    f.write(f"BERT-Score PART B:  {avg_bert_b:.4f}\n")
    f.write(f"BLEU-4 Score:       {avg_bleu:.4f}\n")
    f.write(f"METEOR Score:       {avg_meteor:.4f}\n")
    f.write(f"CLI Score:          {avg_cli:.4f}\n\n")
    f.write("=== DETAILED LOG ===\n")
    f.write("".join(results_log))

print("Report completo salvato in 'eval_report_final.txt'")