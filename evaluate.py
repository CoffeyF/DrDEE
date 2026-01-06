import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from utils import extract_role_level_predictions, compute_role_level_metrics
from postprocess import process_document_predictions
from schema import get_event_role_mapping

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def get_raw_predictions_by_document(config, model, dataset, collate_fn):
    model.eval()
    device = config.device
    all_doc_probs = []
    all_doc_labels = []
    all_doc_event_types = []
    all_doc_info = []
    total_docs = len(dataset)
    with torch.no_grad():
        for doc_idx in range(total_docs):
            batch = [dataset[doc_idx]]
            batch = collate_fn(batch)
            batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            from utils import extract_batch_inputs
            inputs, labels, event_type_id_list = extract_batch_inputs(batch)
            inputs = {k: v for k, v in inputs.items() if k not in ['sens', 'words', 'word_locs']}
            logits = model(**inputs)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            labels_np = labels.cpu().numpy()
            event_types_np = event_type_id_list.cpu().numpy()
            example = dataset.examples[doc_idx]
            num_words = len(example['words'])
            num_roles = len(labels_np) // num_words if num_words > 0 else 0
            all_doc_probs.append(probs)
            all_doc_labels.append(labels_np)
            all_doc_event_types.append(event_types_np)
            event_role_ids = example.get('event_role_ids', [])
            if len(event_role_ids) == 0:
                event_role_ids = inputs.get('event_role_ids', []).cpu().numpy().tolist()
            doc_info = {'words': example['words'], 'event_role_ids': event_role_ids, 'num_words': num_words, 'num_roles': num_roles}
            all_doc_info.append(doc_info)
    return all_doc_probs, all_doc_labels, all_doc_event_types, all_doc_info

def process_predictions_with_threshold(all_doc_probs, all_doc_labels, all_doc_event_types, all_doc_info, threshold=0.5):
    all_doc_predictions = []
    for doc_probs, doc_labels, doc_event_types, doc_info in zip(all_doc_probs, all_doc_labels, all_doc_event_types, all_doc_info):
        num_words = doc_info['num_words']
        num_roles = doc_info['num_roles']
        processed_preds, processed_labs, processed_evts = process_document_predictions(doc_probs, doc_labels, doc_event_types, num_words, num_roles, threshold)
        all_doc_predictions.append(processed_preds)
    return all_doc_predictions

def plot_threshold_analysis(results, best_threshold, best_f1, output_file='threshold_analysis.png'):
    thresholds = [r['threshold'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1_scores = [r['f1'] for r in results]
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2.5, marker='o', markersize=5)
    plt.plot(thresholds, f1_scores, 'g-', label='F1', linewidth=2.5, marker='^', markersize=5)
    plt.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2.5, marker='s', markersize=5)
    plt.axvline(x=best_threshold, color='orange', linestyle='--', linewidth=2, label=f'Best Threshold ({best_threshold:.4f})')
    plt.xlabel('Threshold', fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=14, fontweight='bold')
    plt.title('Precision, F1, Recall vs Threshold (Role Level)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim([min(thresholds), max(thresholds)])
    plt.ylim([0, 0.88])
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

def evaluate_role_level_threshold(config, model, dataset, collate_fn, threshold_start=0.9, threshold_end=0.99, threshold_step=0.01):
    all_doc_probs, all_doc_labels, all_doc_event_types, all_doc_info = get_raw_predictions_by_document(config, model, dataset, collate_fn)
    thresholds = np.arange(threshold_start, threshold_end + threshold_step, threshold_step).tolist()
    thresholds = [t for t in thresholds if t <= threshold_end]
    results = []
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = None
    for idx, threshold in enumerate(thresholds):
        all_doc_predictions = process_predictions_with_threshold(all_doc_probs, all_doc_labels, all_doc_event_types, all_doc_info, threshold)
        total_tp = total_fp = total_fn = 0
        for doc_idx, (doc_preds, doc_labels, doc_event_types, doc_info) in enumerate(zip(all_doc_predictions, all_doc_labels, all_doc_event_types, all_doc_info)):
            pred_roles, gold_roles = extract_role_level_predictions(doc_preds, doc_labels, doc_event_types, doc_info)
            tp, fp, fn = compute_role_level_metrics(pred_roles, gold_roles)
            total_tp += tp
            total_fp += fp
            total_fn += fn
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        results.append({'threshold': threshold, 'tp': total_tp, 'fp': total_fp, 'fn': total_fn, 'precision': precision, 'recall': recall, 'f1': f1})
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {'tp': total_tp, 'fp': total_fp, 'fn': total_fn, 'precision': precision, 'recall': recall, 'f1': f1}
    return results, best_threshold, best_f1, best_metrics

