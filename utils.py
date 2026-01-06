import torch
import numpy as np
from collections import defaultdict
from schema import get_event_role_mapping, get_base_event_type, IDX_TO_EVENT_TYPE

def compute_token_level_metrics(predictions, labels, event_type_ids):
    base_types = set()
    for event_type in IDX_TO_EVENT_TYPE.values():
        base_types.add(get_base_event_type(event_type))
    event_stats = {et: {"TP": 0, "FN": 0, "FP": 0, "TP_FN": 0} for et in base_types}
    for idx, label in enumerate(labels):
        pred = predictions[idx]
        event_type_idx = event_type_ids[idx]
        event_type = IDX_TO_EVENT_TYPE.get(event_type_idx, None)
        if event_type is None:
            continue
        base_type = get_base_event_type(event_type)
        if label == 1:
            event_stats[base_type]["TP_FN"] += 1
            if pred == 1:
                event_stats[base_type]["TP"] += 1
            else:
                event_stats[base_type]["FN"] += 1
        elif label == 0 and pred == 1:
            event_stats[base_type]["FP"] += 1
    result = {}
    total_tp = total_fp = total_fn = 0
    for event_type, stats in event_stats.items():
        precision = stats["TP"] / (stats["TP"] + stats["FP"]) if (stats["TP"] + stats["FP"]) > 0 else 0
        recall = stats["TP"] / (stats["TP"] + stats["FN"]) if (stats["TP"] + stats["FN"]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        result[event_type] = {'pre': precision, 'recall': recall, 'f1': f1, 'TP': stats["TP"], 'FN': stats["FN"], 'FP': stats["FP"], 'TP_FN': stats["TP_FN"]}
        total_tp += stats["TP"]
        total_fp += stats["FP"]
        total_fn += stats["FN"]
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    custom_score = 2 * total_tp / (2 * total_tp + total_fn) if (2 * total_tp + total_fn) > 0 else 0.0
    result['micro'] = {'pre': micro_precision, 'recall': micro_recall, 'f1': micro_f1, 'TP': total_tp, 'FN': total_fn, 'FP': total_fp, 'TP_FN': total_tp + total_fn, 'custom_score': custom_score}
    return result

def extract_batch_inputs(batch):
    return {
        'word_ids': batch[0],
        'wType_ids': batch[1],
        'event_role_ids': batch[3],
    }, batch[2], batch[4]

def compute_role_level_metrics(predicted_roles, gold_roles):
    tp = fp = fn = 0
    all_role_keys = set(predicted_roles.keys()) | set(gold_roles.keys())
    for role_key in all_role_keys:
        pred_tokens = set(predicted_roles.get(role_key, []))
        gold_tokens = set(gold_roles.get(role_key, []))
        if len(gold_tokens) > 0:
            if len(pred_tokens) > 0:
                if pred_tokens == gold_tokens:
                    tp += 1
                else:
                    fp += 1
            else:
                fn += 1
        else:
            if len(pred_tokens) > 0:
                fp += 1
    return tp, fp, fn

def extract_role_level_predictions(document_predictions, document_labels, document_event_types, document_info):
    num_words = document_info['num_words']
    num_roles = document_info['num_roles']
    pred_matrix = document_predictions.reshape(num_words, num_roles)
    label_matrix = document_labels.reshape(num_words, num_roles)
    event_type_matrix = document_event_types.reshape(num_words, num_roles)
    event_role_ids = document_info.get('event_role_ids', [])
    if len(event_role_ids) != num_roles:
        return defaultdict(set), defaultdict(set)
    _, idx_to_event_role = get_event_role_mapping()
    predicted_roles = defaultdict(set)
    gold_roles = defaultdict(set)
    for word_idx in range(num_words):
        for role_idx in range(num_roles):
            pred_val = pred_matrix[word_idx, role_idx]
            label_val = label_matrix[word_idx, role_idx]
            event_role_id = event_role_ids[role_idx]
            if event_role_id == 0:
                continue
            try:
                event_type_name, role_name = idx_to_event_role[event_role_id]
                role_key = (event_type_name, role_name)
                if pred_val == 1:
                    predicted_roles[role_key].add(word_idx)
                if label_val == 1:
                    gold_roles[role_key].add(word_idx)
            except (KeyError, IndexError):
                continue
    return predicted_roles, gold_roles

