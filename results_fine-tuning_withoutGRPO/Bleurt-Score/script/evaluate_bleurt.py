import argparse
import re
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_from_disk
from tqdm import tqdm
import sacrebleu

# --- CONFIGURAZIONE ---
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_PATH = "./output_llama_standard_vsenzaGRPO/final_adapter"
DATASET_PATH = "dataset_final_processed"
EVAL_REPORT_FILE = "eval_report_bleu.txt"


def generate_safe_rule(model, tokenizer, description, justification, label_str, num_beams=1, do_sample=True, temperature=0.1, max_new_tokens=128):
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

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    # configure decoding
    if num_beams and num_beams > 1:
        gen_kwargs.update(dict(num_beams=num_beams, do_sample=False))
    else:
        gen_kwargs.update(dict(do_sample=do_sample, temperature=temperature))

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in generated:
        parts = generated.split("assistant")
        clean_output = parts[-1].strip()
    else:
        clean_output = generated

    return clean_output


def normalize_text(s: str):
    if s is None:
        return ""
    s = s.lower()
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def main(subset: int = 0, beam: int = 1, normalize: bool = False):
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

    total_to_eval = len(test_data) if subset <= 0 else min(subset, len(test_data))
    print(f"Valutazione su {total_to_eval} campioni di test...")
    for i in tqdm(range(total_to_eval)):
        row = test_data[i]
        desc = row['desc']
        just = row['justification']
        lbl = row.get('label', 'Unknown')

        safe_model = generate_safe_rule(
            model, tokenizer, desc, just, str(lbl),
            num_beams=beam, do_sample=(beam <= 1), temperature=0.1
        )
        predictions.append(safe_model)
        references_safe.append(row['safe'])
        references_action.append(row.get('actionTitle', ''))
        references_trigger.append(row.get('triggerTitle', ''))

    # BLEU (corpus-level)
    print("Calcolo BLEU corpus-level in corso...")
    # Optionally normalize texts before scoring
    preds_for_scoring = predictions[:]
    refs_safe_for_scoring = references_safe[:]
    refs_action_for_scoring = references_action[:]
    refs_trigger_for_scoring = references_trigger[:]

    if normalize:
        preds_for_scoring = [normalize_text(p) for p in preds_for_scoring]
        refs_safe_for_scoring = [normalize_text(r) for r in refs_safe_for_scoring]
        refs_action_for_scoring = [normalize_text(r) for r in refs_action_for_scoring]
        refs_trigger_for_scoring = [normalize_text(r) for r in refs_trigger_for_scoring]

    try:
        bleu = sacrebleu.corpus_bleu(preds_for_scoring, [refs_safe_for_scoring])
        score_bleu_pct = bleu.score
    except Exception as e:
        print(f"Errore calcolo BLEU: {e}")
        score_bleu_pct = 0.0

    # BLEU corpus for action and trigger
    try:
        bleu_action = sacrebleu.corpus_bleu(preds_for_scoring, [refs_action_for_scoring])
        score_bleu_action_pct = bleu_action.score
    except Exception:
        score_bleu_action_pct = 0.0

    try:
        bleu_trigger = sacrebleu.corpus_bleu(preds_for_scoring, [refs_trigger_for_scoring])
        score_bleu_trigger_pct = bleu_trigger.score
    except Exception:
        score_bleu_trigger_pct = 0.0

    # Per-sample BLEU counts (soglia 50)
    threshold = 50.0
    num_correct = 0
    num_exact = 0
    per_sample_scores = []
    for pred, ref in zip(preds_for_scoring, refs_safe_for_scoring):
        try:
            s = sacrebleu.sentence_bleu(pred, [ref]).score
        except Exception:
            s = 0.0
        per_sample_scores.append(s)
        if s >= threshold:
            num_correct += 1
        if pred and ref and pred.strip().lower() == ref.strip().lower():
            num_exact += 1

    num_incorrect = len(per_sample_scores) - num_correct

    # Per-sample BLEU for action and trigger
    per_sample_action = []
    per_sample_trigger = []
    num_correct_action = 0
    num_correct_trigger = 0
    for pred, a_ref, t_ref in zip(preds_for_scoring, refs_action_for_scoring, refs_trigger_for_scoring):
        try:
            sa = sacrebleu.sentence_bleu(pred, [a_ref]).score
        except Exception:
            sa = 0.0
        try:
            st = sacrebleu.sentence_bleu(pred, [t_ref]).score
        except Exception:
            st = 0.0
        per_sample_action.append(sa)
        per_sample_trigger.append(st)
        if sa >= threshold:
            num_correct_action += 1
        if st >= threshold:
            num_correct_trigger += 1

    num_incorrect_action = len(per_sample_action) - num_correct_action
    num_incorrect_trigger = len(per_sample_trigger) - num_correct_trigger

    # Percentuali
    total = total_to_eval
    pct_correct = (num_correct / total * 100) if total else 0.0
    pct_correct_action = (num_correct_action / total * 100) if total else 0.0
    pct_correct_trigger = (num_correct_trigger / total * 100) if total else 0.0

    output_txt = (
        f"=== REPORT VALUTAZIONE BLEU ===\n"
        f"Campioni Test: {len(test_data)}\n"
        f"Modello: {ADAPTER_PATH}\n\n"
        f"BLEU (corpus vs safe): {score_bleu_pct:.2f}\n"
        f"BLEU (corpus vs action): {score_bleu_action_pct:.2f}\n"
        f"BLEU (corpus vs trigger): {score_bleu_trigger_pct:.2f}\n\n"
        f"BLEU threshold: {threshold}\n"
        f"Percentuale (BLEU >= {threshold}) vs safe: {pct_correct:.2f}%\n"
        f"Percentuale (BLEU >= {threshold}) vs action: {pct_correct_action:.2f}%\n"
        f"Percentuale (BLEU >= {threshold}) vs trigger: {pct_correct_trigger:.2f}%\n"
    )

    print(output_txt)
    with open(EVAL_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(output_txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BLEU with optional decoding and normalization")
    parser.add_argument("--subset", type=int, default=0, help="Number of test samples to evaluate (0 = all)")
    parser.add_argument("--beam", type=int, default=1, help="Number of beams for beam search (1 = no beam, i.e., sampling)")
    parser.add_argument("--normalize", action="store_true", help="Normalize text (lowercase/whitespace) before scoring")
    args = parser.parse_args()
    main(subset=args.subset, beam=args.beam, normalize=args.normalize)


if __name__ == "__main__":
    main()
