import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_from_disk
from tqdm import tqdm

try:
    from nltk.translate.meteor_score import meteor_score
    import nltk
except Exception:
    meteor_score = None

# --- CONFIGURAZIONE ---
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_PATH = "./output_llama_standard_vsenzaGRPO/final_adapter"
DATASET_PATH = "dataset_final_processed"
EVAL_REPORT_FILE = "eval_report_meteor.txt"


def generate_safe_rule(model, tokenizer, description, justification, label_str, max_new_tokens=128):
    user_msg = (
        f"DESCRIPTION: {description}\n"
        f"JUSTIFICATION: {justification}\n"
        f"LABEL: {label_str}"
    )

    input_text = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are an expert AI security assistant. Your task is to generate a 'safe' automation rule based on the description, justification, and risk category provided.\n"
        "Input Format:\n"
        "DESCRIPTION: <description of the rule>\n"
        "JUSTIFICATION: <security justification>\n"
        "LABEL: <risk category>\n\n"
        "Output:\n"
        "SAFE_MODEL: <safe automation rule><|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in generated:
        parts = generated.split("assistant")
        clean_output = parts[-1].strip()
    else:
        clean_output = generated

    return clean_output


def ensure_nltk():
    # tenta di scaricare risorse necessarie a METEOR se mancanti
    global meteor_score
    if meteor_score is None:
        try:
            import nltk
            nltk.download('wordnet', quiet=True)
            from nltk.translate.meteor_score import meteor_score as ms
            meteor_score = ms
        except Exception:
            meteor_score = None


def main():
    ensure_nltk()

    print("Caricamento Tokenizer e Modello...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    if os.path.exists(ADAPTER_PATH):
        print(f"Caricamento Adapter da {ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        model = model.merge_and_unload()
    else:
        print(f"⚠️ ATTENZIONE: Adapter non trovato in {ADAPTER_PATH}. Uso il modello base.")

    model.eval()

    if not os.path.exists(DATASET_PATH):
        print(f"Errore: Dataset {DATASET_PATH} non trovato.")
        return

    dataset = load_from_disk(DATASET_PATH)
    test_data = dataset['test']

    predictions = []
    references_safe = []
    references_action = []
    references_trigger = []

    print(f"Valutazione su {len(test_data)} campioni di test...")
    for i in tqdm(range(len(test_data))):
        row = test_data[i]
        desc = row['desc']
        just = row['justification']
        lbl = row.get('label', 'Unknown')

        safe_model = generate_safe_rule(model, tokenizer, desc, just, str(lbl), max_new_tokens=128)
        predictions.append(safe_model)
        references_safe.append(row['safe'])
        references_action.append(row.get('actionTitle', ''))
        references_trigger.append(row.get('triggerTitle', ''))

    if meteor_score is None:
        print("METEOR non disponibile (nltk o risorse mancanti). Installare 'nltk' e scaricare 'wordnet'.")
        avg_meteor = 0.0
        per_sample = [0.0] * len(predictions)
        per_sample_action = [0.0] * len(predictions)
        per_sample_trigger = [0.0] * len(predictions)
    else:
        print("Calcolo METEOR per-sample in corso...")
        per_sample = []
        per_sample_action = []
        per_sample_trigger = []
        for pred, ref in zip(predictions, references_safe):
            try:
                # meteor_score expects references as list
                s = meteor_score([ref], pred)
            except Exception:
                s = 0.0
            per_sample.append(s)
        # calcola anche rispetto ad action e trigger
        for pred, a_ref in zip(predictions, references_action):
            try:
                sa = meteor_score([a_ref], pred)
            except Exception:
                sa = 0.0
            per_sample_action.append(sa)
        for pred, t_ref in zip(predictions, references_trigger):
            try:
                st = meteor_score([t_ref], pred)
            except Exception:
                st = 0.0
            per_sample_trigger.append(st)
        try:
            avg_meteor = sum(per_sample) / len(per_sample) if per_sample else 0.0
        except Exception:
            avg_meteor = 0.0

    avg_meteor_pct = avg_meteor * 100

    # Conteggi (soglia 0.5)
    threshold = 0.5
    num_correct = sum(1 for s in per_sample if s >= threshold)
    num_incorrect = len(per_sample) - num_correct
    num_correct_action = sum(1 for s in per_sample_action if s >= threshold)
    num_incorrect_action = len(per_sample_action) - num_correct_action
    num_correct_trigger = sum(1 for s in per_sample_trigger if s >= threshold)
    num_incorrect_trigger = len(per_sample_trigger) - num_correct_trigger

    # Exact match
    num_exact = sum(1 for p, r in zip(predictions, references_safe) if p and r and p.strip().lower() == r.strip().lower())

    total = len(test_data)
    pct_correct = (num_correct / total * 100) if total else 0.0
    pct_correct_action = (num_correct_action / total * 100) if total else 0.0
    pct_correct_trigger = (num_correct_trigger / total * 100) if total else 0.0

    output_txt = (
        f"=== REPORT VALUTAZIONE METEOR ===\n"
        f"Campioni Test: {len(test_data)}\n"
        f"Modello: {ADAPTER_PATH}\n\n"
        f"METEOR (avg vs safe): {avg_meteor_pct:.2f}%\n\n"
        f"METEOR threshold: {threshold}\n"
        f"Percentuale (METEOR >= {threshold}) vs safe: {pct_correct:.2f}%\n"
        f"Percentuale (METEOR >= {threshold}) vs action: {pct_correct_action:.2f}%\n"
        f"Percentuale (METEOR >= {threshold}) vs trigger: {pct_correct_trigger:.2f}%\n"
    )

    print(output_txt)
    with open(EVAL_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(output_txt)

    # --- DEBUG: dump per-sample pairs and simple flags ---
    debug_file = "meteor_pairs_debug.txt"
    max_tokens = 128
    with open(debug_file, "w", encoding="utf-8") as df:
        for i, (pred, ref) in enumerate(zip(predictions, references_safe)):
            pred_str = pred if pred is not None else ""
            ref_str = ref if ref is not None else ""
            empty_pred = True if not pred_str.strip() else False
            exact_match = False
            try:
                exact_match = pred_str.strip().lower() == ref_str.strip().lower()
            except Exception:
                exact_match = False
            # detect truncation heuristically using tokenizer length
            try:
                toks = tokenizer.encode(pred_str, add_special_tokens=False)
                truncated = len(toks) >= (max_tokens - 4)
            except Exception:
                truncated = False

            df.write(f"INDEX: {i}\n")
            df.write(f"REF_SAFE: {ref_str}\n")
            df.write(f"PRED: {pred_str}\n")
            df.write(f"EMPTY_PRED: {empty_pred}\n")
            df.write(f"EXACT_MATCH: {exact_match}\n")
            df.write(f"TRUNCATED_HEURISTIC: {truncated}\n")
            df.write("---\n")
    print(f"Debug dump scritto in {debug_file}")


if __name__ == "__main__":
    main()
