import torch
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
from utils import extract_batch_inputs, compute_token_level_metrics

def find_continuous_token_groups(indices):
    if len(indices) == 0:
        return []
    groups = []
    current_group = [indices[0]]
    for i in range(1, len(indices)):
        if indices[i] == indices[i-1] + 1:
            current_group.append(indices[i])
        else:
            groups.append(current_group)
            current_group = [indices[i]]
    groups.append(current_group)
    return groups

def resolve_column_conflicts_with_threshold(column_predictions, threshold=0.5):
    high_prob_indices = np.where(column_predictions > threshold)[0]
    if len(high_prob_indices) == 0:
        return np.zeros_like(column_predictions)
    continuous_groups = find_continuous_token_groups(high_prob_indices)
    result = np.zeros_like(column_predictions)
    candidates = []
    for group in continuous_groups:
        if len(group) == 1:
            candidates.append((group[0], column_predictions[group[0]]))
        else:
            group_probs = column_predictions[group]
            max_prob = np.max(group_probs)
            min_prob = np.min(group_probs)
            if max_prob - min_prob <= 0.8:
                candidates.append((group, max_prob))
            else:
                for idx in group:
                    candidates.append((idx, column_predictions[idx]))
    if candidates:
        best_candidate, best_prob = max(candidates, key=lambda x: x[1])
        if isinstance(best_candidate, (int, np.integer)):
            result[best_candidate] = 1
        else:
            for idx in best_candidate:
                result[idx] = 1
    return result

def process_document_predictions(document_predictions, document_labels, document_event_types, num_words, num_roles, threshold=0.5):
    pred_matrix = document_predictions.reshape(num_words, num_roles)
    label_matrix = document_labels.reshape(num_words, num_roles)
    event_type_matrix = document_event_types.reshape(num_words, num_roles)
    processed_matrix = np.zeros_like(pred_matrix)
    for col in range(num_roles):
        column_predictions = pred_matrix[:, col]
        processed_column = resolve_column_conflicts_with_threshold(column_predictions, threshold)
        processed_matrix[:, col] = processed_column
    processed_preds = processed_matrix.flatten()
    processed_labels = label_matrix.flatten()
    processed_event_types = event_type_matrix.flatten()
    return processed_preds, processed_labels, processed_event_types

def get_model_predictions_by_document(config, model, dataset, collate_fn, output_file):
    model.eval()
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=config.per_gpu_eval_batch_size, collate_fn=collate_fn)
    all_predictions = []
    all_labels = []
    all_event_type_ids = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            if batch[0].numel() == 0:
                continue
            batch = tuple(t.to(config.device) if isinstance(t, torch.Tensor) else t for t in batch)
            inputs, labels, event_type_id_list = extract_batch_inputs(batch)
            if labels.dtype != torch.long:
                continue
            logits = model(**inputs)
            probs = torch.softmax(logits, dim=1)
            positive_probs = probs[:, 1].cpu().numpy()
            word_count = len(inputs['word_ids'])
            if len(positive_probs) % word_count == 0:
                num_roles = len(positive_probs) // word_count
            else:
                continue
            if num_roles == 0:
                continue
            pred_matrix = positive_probs.reshape(word_count, num_roles)
            label_matrix = labels.cpu().numpy().reshape(word_count, num_roles)
            event_type_matrix = event_type_id_list.cpu().numpy().reshape(word_count, num_roles)
            processed_preds, processed_labels, processed_event_types = process_document_predictions(positive_probs, labels.cpu().numpy(), event_type_id_list.cpu().numpy(), word_count, num_roles)
            all_predictions.extend(processed_preds)
            all_labels.extend(processed_labels)
            all_event_type_ids.extend(processed_event_types)
    predictions_array = np.array(all_predictions).astype(int)
    np.save(f"{output_file}_predictions.npy", predictions_array)
    np.save(f"{output_file}_labels.npy", np.array(all_labels))
    np.save(f"{output_file}_event_types.npy", np.array(all_event_type_ids))
    return predictions_array, np.array(all_labels), np.array(all_event_type_ids)

def evaluate_with_conflict_resolution(config, model, dataset, collate_fn, output_file=None, quiet=True):
    import tempfile
    import os
    import sys
    from io import StringIO
    if quiet:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
    try:
        if output_file is None:
            output_file = os.path.join(tempfile.gettempdir(), f'temp_eval_{os.getpid()}')
        predictions, labels, event_types = get_model_predictions_by_document(config, model, dataset, collate_fn, output_file)
        results = compute_token_level_metrics(predictions, labels, event_types)
        custom_score = results.get('micro', {}).get('custom_score', 0.0)
        if output_file.startswith(tempfile.gettempdir()):
            try:
                os.remove(f"{output_file}_predictions.npy")
                os.remove(f"{output_file}_labels.npy")
                os.remove(f"{output_file}_event_types.npy")
            except:
                pass
        return custom_score, results
    finally:
        if quiet:
            sys.stdout = old_stdout

