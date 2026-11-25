from __future__ import annotations
import argparse, hashlib, json, os, random, re, struct, zlib
from pathlib import Path
from typing import Any
from evaluation_core import load_policy, parse_integer, qwen_call, read_json, read_jsonl, sha256_file
def inventory(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob('*')):
        if path.is_file() and (not path.is_symlink()):
            rows.append({'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)})
    return rows

def load_update_examples(update_root: Path, image_root: Path) -> list[dict[str, Any]]:
    manifest = read_json(update_root / 'update_manifest.json')
    image_root = image_root.resolve(strict=True)
    if manifest.get('schema') != 'rsi_visual_verifier_update_set_v2' or manifest.get('status') != 'ACCEPTED':
        raise ValueError('update manifest is stale or has an unexpected schema')
    if manifest.get('image_root') != str(image_root):
        raise ValueError('update manifest points to another image root')
    if manifest.get('evaluator_answers_in_update_set') is not False:
        raise ValueError('update manifest does not prove claim-only inputs')
    examples_path = update_root / 'update_examples.jsonl'
    records = [row for row in manifest.get('files', []) if row.get('path') == 'update_examples.jsonl']
    if len(records) != 1 or records[0].get('sha256') != sha256_file(examples_path) or records[0].get('bytes') != examples_path.stat().st_size:
        raise ValueError('update examples do not match the sealed manifest file digest')
    examples = read_jsonl(examples_path)
    if len(examples) != manifest.get('fixed_budget') or len(examples) != manifest.get('selected_count'):
        raise ValueError('update example count does not match its manifest')
    scene_ids: set[str] = set()
    for example in examples:
        required = {'sample_id', 'image_path', 'question', 'claimed_answer', 'source_scene_id', 'image_sha256'}
        if set(example) != required:
            raise ValueError(f"update example has unexpected or missing fields: {example.get('sample_id')}")
        image = (image_root / str(example['image_path'])).resolve(strict=True)
        if image_root not in image.parents or not image.is_file() or image.is_symlink():
            raise ValueError(f"update image escapes dataset root: {example['sample_id']}")
        if not isinstance(example.get('claimed_answer'), str) or not re.fullmatch('[0-4]', example['claimed_answer']):
            raise ValueError(f"invalid claim target: {example['sample_id']}")
        if not isinstance(example.get('source_scene_id'), str) or example['source_scene_id'] in scene_ids:
            raise ValueError(f"update examples reuse a source scene: {example['sample_id']}")
        scene_ids.add(example['source_scene_id'])
        if sha256_file(image) != example['image_sha256']:
            raise ValueError(f"update image digest differs from manifest: {example['sample_id']}")
    return examples

def make_supervised_inputs(processor: Any, model: Any, image_path: Path, question: str, answer: str):
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    user = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': f'{question} Return only the integer, with no other words.'}]}]
    full = user + [{'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]}]
    prompt_text = processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
    prompt_inputs = processor(text=[prompt_text], images=[image], return_tensors='pt', padding=True)
    inputs = processor(text=[full_text], images=[image], return_tensors='pt', padding=True)
    labels = inputs['input_ids'].clone()
    prompt_length = prompt_inputs['input_ids'].shape[1]
    if prompt_length >= labels.shape[1]:
        raise ValueError('assistant answer produced no trainable tokens')
    labels[:, :prompt_length] = -100
    pad_id = getattr(processor.tokenizer, 'pad_token_id', None)
    if pad_id is not None:
        labels[labels == pad_id] = -100
    image_token_id = getattr(getattr(model, 'config', None), 'image_token_id', None)
    if isinstance(image_token_id, int):
        labels[labels == image_token_id] = -100
    supervised_text = processor.tokenizer.decode(labels[0][labels[0] != -100].tolist(), skip_special_tokens=True).strip()
    if supervised_text != answer:
        raise ValueError(f'supervised tokens do not recover the claimed answer: {supervised_text!r} vs {answer!r}')
    inputs['labels'] = labels
    return (image, inputs)

def target_modules(model: Any) -> list[str]:
    import torch
    suffixes = {name.rsplit('.', 1)[-1] for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    preferred = [name for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj') if name in suffixes]
    if len(preferred) < 2:
        raise RuntimeError(f'could not find stable attention LoRA targets; found {sorted(suffixes)[:40]}')
    return preferred

def training_order(examples: list[dict[str, Any]], seed: int, max_steps: int, epochs: int, require_complete_epoch: bool=False) -> list[dict[str, Any]]:
    """Resolve the consumed-example schedule before model allocation."""
    if not examples or max_steps < 1 or epochs < 1:
        raise ValueError('training needs examples and positive steps and epochs')
    if require_complete_epoch and max_steps < len(examples):
        raise ValueError('step cap cannot consume one complete selected-example epoch')
    ordered = list(examples)
    random.Random(seed).shuffle(ordered)
    return ordered

def train_adapter(model: Any, processor: Any, examples: list[dict[str, Any]], image_root: Path, output_dir: Path, args: argparse.Namespace):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    modules = target_modules(model)
    config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias='none', task_type=TaskType.CAUSAL_LM, target_modules=modules)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    losses: list[float] = []
    steps = 0
    ordered_examples = training_order(examples, args.seed, args.max_steps, args.epochs, getattr(args, 'require_complete_epoch', False))
    consumed_ids: list[str] = []
    for epoch in range(args.epochs):
        for example in ordered_examples:
            image, inputs = make_supervised_inputs(processor, model, (image_root / example['image_path']).resolve(strict=True), str(example['question']), str(example['claimed_answer']))
            inputs = {key: value.to('cuda') if hasattr(value, 'to') else value for key, value in inputs.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**inputs)
            loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite training loss at step {steps + 1}: {loss.item()}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            consumed_ids.append(str(example['sample_id']))
            losses.append(float(loss.detach().cpu()))
            steps += 1
            print(f'TRAIN_PROGRESS={steps}/{args.max_steps} LOSS={losses[-1]:.6f}', flush=True)
            del image, inputs, output, loss
            if steps >= args.max_steps:
                break
        if steps >= args.max_steps:
            break
    if steps == 0:
        raise RuntimeError('no training steps completed')
    output_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    consumed_unique = set(consumed_ids)
    complete_epochs = steps // len(ordered_examples)
    if getattr(args, 'require_complete_epoch', False) and consumed_unique != {str(e['sample_id']) for e in examples}:
        raise RuntimeError('completed training did not consume every selected example')
    return ({'steps': steps, 'epochs_completed': complete_epochs, 'examples_available': len(examples), 'examples_seen': len(consumed_unique), 'example_presentations': len(consumed_ids), 'consumed_sample_ids': consumed_ids, 'complete_selected_epoch': complete_epochs >= 1, 'order_seed': args.seed, 'loss_first': losses[0], 'loss_last': losses[-1], 'loss_min': min(losses), 'target_modules': modules, 'lora_rank': args.lora_rank, 'lora_alpha': args.lora_alpha, 'learning_rate': args.learning_rate, 'adapter_inventory': inventory(output_dir)}, model)

def evaluate(model: Any, processor: Any, dataset_root: Path, items_per_split: int, prediction_output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy = load_policy(dataset_root, expected_items_per_split=items_per_split, selected_splits=('heldout',))
    heldout = [row for row in policy if row['split'] == 'heldout']
    predictions = []
    for row in heldout:
        text = qwen_call(model, processor, Path(row['image_path']), f"{row['question']} Return only the integer, with no other words.")
        predictions.append({'sample_id': row['sample_id'], 'prediction_text': text, 'prediction': parse_integer(text)})
    with prediction_output.open('x', encoding='utf-8') as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    prediction_sha256 = sha256_file(prediction_output)
    labels = {str(label['sample_id']): label for label in read_jsonl(dataset_root / 'evaluator_labels' / 'heldout.jsonl')}
    for row in predictions:
        label = labels[row['sample_id']]
        row['ground_truth_answer'] = str(label['answer'])
        row['correct'] = row['prediction'] is not None and str(row['prediction']) == str(label['answer'])
        row['edit_kind'] = label['edit_kind']
    summary = {'count': len(predictions), 'accuracy': sum((row['correct'] for row in predictions)) / len(predictions) if predictions else None, 'by_edit_kind': {kind: {'count': sum((row['edit_kind'] == kind for row in predictions)), 'accuracy': sum((row['correct'] for row in predictions if row['edit_kind'] == kind)) / sum((row['edit_kind'] == kind for row in predictions)) if any((row['edit_kind'] == kind for row in predictions)) else None} for kind in ('clean', 'answer_changing', 'answer_preserving')}}
    summary['label_free_predictions_sha256'] = prediction_sha256
    summary['labels_opened_after_prediction_freeze'] = True
    return (predictions, summary)

def load_qwen(model_dir: Path):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16, device_map={'': 0}, local_files_only=True, use_safetensors=True).eval()
    return (model, processor)
