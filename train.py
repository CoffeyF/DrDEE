import random
import os
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import trange, tqdm
from utils import extract_batch_inputs, compute_token_level_metrics
from postprocess import evaluate_with_conflict_resolution

torch.set_printoptions(profile="full")

def set_random_seed(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

def train_model(config, model, train_dataset, dev_dataset, test_dataset, train_label_weights, dev_label_weights, test_label_weights, collate_fn):
    writer = SummaryWriter()
    config.train_batch_size = config.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config.train_batch_size, collate_fn=collate_fn)
    if config.max_steps > 0:
        total_steps = config.max_steps
        config.num_train_epochs = config.max_steps // (len(train_dataloader) // config.gradient_accumulation_steps) + 1
    else:
        total_steps = len(train_dataloader) // config.gradient_accumulation_steps * config.num_train_epochs
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=config.learning_rate)
    global_step = 0
    train_loss = logging_loss = 0.0
    model.zero_grad()
    train_iterator = trange(int(config.num_train_epochs), desc="Epoch")
    set_random_seed(config)
    epoch = 0
    train_label_weights = torch.tensor(train_label_weights).to(config.device)
    dev_label_weights = torch.tensor(dev_label_weights).to(config.device)
    best_custom_score = 0.0
    patience = 5
    patience_counter = 0
    best_model_path = os.path.join(config.output_dir, 'best_model.pth')
    os.makedirs(config.output_dir, exist_ok=True)
    result_file = open('./output/result.txt', 'w', encoding='utf-8')
    for epoch_idx in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch_idx+1}/{int(config.num_train_epochs)}", leave=False, position=1)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            if batch[0].numel() == 0:
                continue
            batch = tuple(t.to(config.device) if isinstance(t, torch.Tensor) else t for t in batch)
            inputs, labels, event_type_ids = extract_batch_inputs(batch)
            if labels.dtype != torch.long:
                continue
            logits = model(**inputs)
            loss = F.cross_entropy(logits, labels, weight=train_label_weights)
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps
            loss.backward()
            train_loss += loss.item()
            epoch_iterator.set_postfix({'loss': f'{loss.item():.4f}'})
            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if config.logging_steps > 0 and global_step % config.logging_steps == 0:
                    writer.add_scalar('train_loss', (train_loss - logging_loss) / config.logging_steps, global_step)
                    logging_loss = train_loss
        eval_results, eval_loss = evaluate_model(config, dev_dataset, model, dev_label_weights, collate_fn, result_file)
        writer.add_scalar('train_epoch_loss', (train_loss - logging_loss) / config.logging_steps, epoch)
        custom_score, _ = evaluate_with_conflict_resolution(config, model, dev_dataset, collate_fn)
        current_f1 = eval_results.get('micro', {}).get('f1', 0.0)
        if custom_score > best_custom_score:
            best_custom_score = custom_score
            patience_counter = 0
            torch.save({'epoch': epoch_idx, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_custom_score': best_custom_score, 'config': config}, best_model_path)
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
        epoch += 1
    writer.close()
    checkpoint = torch.load(best_model_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_results, test_loss = evaluate_model(config, test_dataset, model, test_label_weights, collate_fn, result_file)
    result_file.close()

def evaluate_model(config, eval_dataset, model, label_weights, collate_fn, result_file):
    config.eval_batch_size = config.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=config.eval_batch_size, collate_fn=collate_fn)
    label_weights = torch.tensor(label_weights).to(config.device)
    eval_loss = 0.0
    num_eval_steps = 0
    all_predictions = []
    all_labels = []
    all_event_type_ids = []
    eval_iterator = tqdm(eval_dataloader, desc="Evaluating", leave=False)
    for batch in eval_iterator:
        if batch[0].numel() == 0:
            continue
        model.eval()
        batch = tuple(t.to(config.device) if isinstance(t, torch.Tensor) else t for t in batch)
        inputs, labels, event_type_id_list = extract_batch_inputs(batch)
        if labels.dtype != torch.long:
            continue
        logits = model(**inputs)
        loss = F.cross_entropy(logits, labels, weight=label_weights)
        eval_loss += loss.mean().item()
        num_eval_steps += 1
        predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
        all_predictions += predictions.tolist()
        all_labels += labels.detach().cpu().tolist()
        all_event_type_ids += event_type_id_list.detach().cpu().tolist()
        eval_iterator.set_postfix({'eval_loss': f'{loss.mean().item():.4f}'})
    eval_loss = eval_loss / num_eval_steps
    results = compute_token_level_metrics(all_predictions, all_labels, all_event_type_ids)
    from schema import IDX_TO_EVENT_TYPE, get_base_event_type
    processed_base_types = set()
    for event_type_idx, event_type in IDX_TO_EVENT_TYPE.items():
        base_event_type = get_base_event_type(event_type)
        if base_event_type in results and base_event_type not in processed_base_types:
            for key, value in results[base_event_type].items():
                result_file.write(key + '=' + str(value) + '\n')
            processed_base_types.add(base_event_type)
    if 'micro' in results:
        for key, value in results['micro'].items():
            result_file.write('micro_' + key + '=' + str(value) + '\n')
    return results, eval_loss

