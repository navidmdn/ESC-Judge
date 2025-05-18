from unsloth import FastLanguageModel, get_chat_template
from fire import Fire
import json
from tqdm import tqdm
import re
from typing import Sequence, Dict, Any, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def measure_template_following_accuracy(predictions: List[str]):
    correct = 0
    for pred in predictions:
        if _is_following_template(pred):
            correct += 1
    return {'correct_template': (correct / len(predictions))}


def _is_following_template(response_txt: str):
    match = re.findall("## Reasoning\n.*## Verdict\n.*", response_txt)
    if match:
        return True
    return False

def measure_verdict_matching_stats(predictions: List[str], references: List[str]):

    cat_to_id = {
        'a': 0,
        'b': 1,
        'tie': 2
    }

    pred_classes = []
    ref_classes = []
    bad_format_cnt = 0

    for pred, ref in zip(predictions, references):
        try:
            pred_verdict = _get_verdict_(pred)
            ref_verdict = _get_verdict_(ref)
            pred_classes.append(cat_to_id[pred_verdict])
            ref_classes.append(cat_to_id[ref_verdict])

        except Exception:
            bad_format_cnt += 1
            continue

    sorted_labels: List[int] = sorted(cat_to_id.values())
    id2label: Dict[int, str] = {v: k for k, v in cat_to_id.items()}

    prec, rec, f1, support = precision_recall_fscore_support(
        ref_classes,
        pred_classes,
        labels=sorted_labels,
        average=None,
        zero_division=0,  # avoids division‑by‑zero warnings for rare classes
    )

    metrics = {
        "accuracy": accuracy_score(ref_classes, pred_classes),
        "precision": {id2label[i]: p for i, p in zip(sorted_labels, prec)},
        "recall": {id2label[i]: r for i, r in zip(sorted_labels, rec)},
        "f1": {id2label[i]: f for i, f in zip(sorted_labels, f1)},
        "support": {id2label[i]: s for i, s in zip(sorted_labels, support)},
        "total": len(predictions),
        "bad_format": bad_format_cnt,
    }
    return metrics



def _get_verdict_(output_txt: str):
    try:
        winner_text = output_txt.split("# Verdict")[1].strip().lower()
    except Exception as e:
        # print(f"bad formatting of verdict detected__ NO VERDICT!: {output_txt}")
        raise e

    assert "model a" in winner_text or "model b" in winner_text or "tie" in winner_text, f"Bad verdict formatting"

    match1 = re.search(r"\*\*(?:support\s+)?model\s+([ab])\*\* (demonstrates a slightly higher|performs better)",
                       winner_text)
    match2 = re.search(r"\*\*(?:support\s+)?model\s+([ab])\*\* outperforms", winner_text)

    if "tie" in winner_text:
        return 'tie'
    elif 'model a' in winner_text and 'model b' not in winner_text:
        return 'a'
    elif 'model b' in winner_text and 'model a' not in winner_text:
        return 'b'
    elif match1:
        return match1.group(1)
    elif match2:
        return match2.group(1)
    else:
        # print(f"bad formatting of verdict detected. Couldn't parse verdict: {winner_text}")
        raise Exception(f"Couldn't parse verdict")

def load_jsonl(path):
    res = []
    with open(path) as f:
        for line in f:
            res.append(json.loads(line))
    return res

def get_response_references(batch, model, tokenizer):
    inputs = []
    references = []

    for item in batch:
        inputs.append(item['messages'][:2])
        references.append(item['messages'][2]['content'])

    tokenized_inputs = tokenizer.apply_chat_template(
        inputs,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding="longest",
    ).to(model.device)

    outputs = model.generate(
        input_ids=tokenized_inputs,
        max_new_tokens=1024,
        use_cache=True,
        temperature=1.0,
    )

    responses = tokenizer.batch_decode(outputs[:, tokenized_inputs.shape[-1]:], skip_special_tokens=True)
    print(responses[0])
    return responses, references


def evaluate(
        model_path,
        output_file='metrics.json',
        evaluation_data_path='./data/test.json',
        max_seq_length=12000,
        load_in_4bit=True,
        dtype=None,
        cache_dir=None,
        inference_batch_size=1,
):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=dtype,
        cache_dir=cache_dir,
        fast_inference=True
    )

    # todo: build it according to the model
    if 'llama-3.1' in model_path.lower():
        chat_template_name = 'llama-3.1'
    elif 'llama-3.2' in model_path.lower():
        chat_template_name = 'llama-3.2'
    else:
        print("WARNING: Using default chat template: llama3.1")
        chat_template_name = 'llama-3.1'


    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template_name,
    )

    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    eval_data = load_jsonl(evaluation_data_path)

    responses = []
    references = []

    for batch_idx in tqdm(range(0, len(eval_data), inference_batch_size)):
        batch = eval_data[batch_idx:batch_idx+inference_batch_size]
        batch_responses, batch_references = get_response_references(batch, model, tokenizer)
        responses.extend(batch_responses)
        references.extend(batch_references)


    metrics = {}
    template_metrics = measure_template_following_accuracy(responses)
    verdict_metrics = measure_verdict_matching_stats(responses, references)
    metrics.update(template_metrics)
    metrics.update(verdict_metrics)

    with open(output_file, 'w') as f:
        json.dump(metrics, f)



if __name__ == "__main__":
    Fire(evaluate)