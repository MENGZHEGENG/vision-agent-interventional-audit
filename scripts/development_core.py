from __future__ import annotations
import argparse, hashlib, json, os, random, re, struct, zlib
from pathlib import Path
from typing import Any

MODEL_PANEL = {'qwen3_vl_2b_instruct': {'repo_id': 'Qwen/Qwen3-VL-2B-Instruct', 'revision': '89644892e4d85e24eaac8bacfd4f463576704203', 'family': 'qwen3_vl'}, 'qwen3_vl_8b_instruct': {'repo_id': 'Qwen/Qwen3-VL-8B-Instruct', 'revision': '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b', 'family': 'qwen3_vl'}, 'internvl3_5_2b_instruct': {'repo_id': 'OpenGVLab/InternVL3_5-2B-Instruct', 'revision': 'a1e50c526faea805730148e7b1f2af18abe9e4b2', 'family': 'internvl3_5'}, 'internvl3_5_8b': {'repo_id': 'OpenGVLab/InternVL3_5-8B', 'revision': '9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79', 'family': 'internvl3_5'}}

SHA256_RE = re.compile('[0-9a-f]{64}\\Z')

ANSWER_RE = re.compile('[0-4]\\Z')

GENERATION = {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 16}

def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')

def read_json(path: Path) -> Any:
    return json.loads(path.read_text('utf-8'))

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_bytes().splitlines(), 1):
        row = json.loads(line.decode('utf-8'))
        if not isinstance(row, dict):
            raise ValueError(f'non-object JSONL row {number}: {path}')
        rows.append(row)
    return rows

def parse_answer(raw: str) -> tuple[str | None, str]:
    candidate = raw.strip()
    if ANSWER_RE.fullmatch(candidate):
        return (candidate, 'PARSEABLE')
    return (None, 'INVALID_FORMAT')

def parse_verdict(raw: str) -> tuple[str | None, str]:
    verdict = raw.strip()
    if verdict in {'YES', 'NO'}:
        return (verdict, 'PARSEABLE')
    return (None, 'UNPARSEABLE_REJECTION')

def validate_model_input_metadata(family: str, image_present: bool, metadata: dict[str, Any]) -> None:
    if not isinstance(metadata.get('prompt_sha256'), str) or not SHA256_RE.fullmatch(metadata['prompt_sha256']):
        raise ValueError('model call did not record a prompt digest')
    if family == 'qwen3_vl':
        shapes = metadata.get('tensor_shapes')
        if not isinstance(shapes, dict) or 'input_ids' not in shapes:
            raise ValueError('Qwen model call did not expose token input shapes')
        has_image_tensor = any((key in shapes for key in ('pixel_values', 'image_grid_thw')))
        if image_present != has_image_tensor:
            raise ValueError('Qwen model input image-tensor presence differs from the assigned condition')
    elif family == 'internvl3_5':
        shapes = metadata.get('generation_input_shapes')
        if not isinstance(shapes, dict) or not shapes.get('input_ids') or (not shapes.get('attention_mask')):
            raise ValueError('InternVL model.generate did not receive token tensors')
        has_image_tensor = shapes.get('pixel_values') is not None
        if image_present != has_image_tensor:
            raise ValueError('InternVL model.generate image tensor differs from the assigned condition')
        if image_present and shapes['pixel_values'] != metadata.get('pixel_values_shape'):
            raise ValueError('InternVL chat image tensor changed before model.generate')
    else:
        raise ValueError(f'unsupported model family: {family}')

def build_text(stage: str, question: str, answer: str | None) -> str:
    if stage == 'candidate':
        return 'Answer the visual counting question. Count the objects that meet the description in the question. Respond with exactly one integer from 0 to 4 and no other text.\nQuestion: ' + question
    return f'Check the candidate answer against the question and the available image. Treat the candidate as a claim to verify. Respond with exactly YES if it is correct or exactly NO if it is incorrect. Do not explain.\nQuestion: {question}\nCandidate answer: {answer}'

def load_model(model_key: str, model_dir: Path):
    import torch
    from PIL import Image
    spec = MODEL_PANEL[model_key]
    if spec['family'] == 'qwen3_vl':
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(str(model_dir), dtype=torch.bfloat16, device_map={'': 0}, local_files_only=True, use_safetensors=True).eval()
        return (model, processor, None)
    from torchvision import transforms
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(str(model_dir), dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=False, trust_remote_code=True, local_files_only=True, device_map={'': 0}).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, use_fast=False, local_files_only=True)
    transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return (model, tokenizer, transform)

def qwen_call(torch, model, processor, image, text: str) -> tuple[str, dict[str, Any]]:
    content = []
    if image is not None:
        content.append({'type': 'image', 'image': image})
    content.append({'type': 'text', 'text': text})
    messages = [{'role': 'user', 'content': content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    kwargs: dict[str, Any] = {'text': [prompt], 'return_tensors': 'pt', 'padding': True}
    if image is not None:
        kwargs['images'] = [image]
    inputs = processor(**kwargs)
    if image is None and any((key in inputs for key in ('pixel_values', 'image_grid_thw'))):
        raise ValueError('Qwen emitted image tensors for the no-image input')
    if image is not None and (not any((key in inputs for key in ('pixel_values', 'image_grid_thw')))):
        raise ValueError('Qwen processor omitted image tensors for an image input')
    tensor_shapes = {key: list(value.shape) for key, value in inputs.items() if hasattr(value, 'shape')}
    inputs = {key: value.to('cuda:0') if hasattr(value, 'to') else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=GENERATION['max_new_tokens'], do_sample=False, num_beams=1)
    prompt_length = inputs['input_ids'].shape[1]
    response = processor.batch_decode(generated[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return (response, {'prompt_sha256': sha256_bytes(prompt.encode()), 'tensor_shapes': tensor_shapes, 'input_tensor_keys': sorted(tensor_shapes)})

def intern_call(torch, model, tokenizer, transform, image, text: str) -> tuple[str, dict[str, Any]]:
    if image is None:
        prompt = text
        pixels = None
        shape = None
    else:
        pixels = transform(image).unsqueeze(0).to(dtype=torch.bfloat16, device='cuda:0')
        prompt = '<image>\n' + text
        shape = list(pixels.shape)
    config = {'max_new_tokens': GENERATION['max_new_tokens'], 'do_sample': False, 'num_beams': 1}
    captured: dict[str, list[int] | None] = {}
    original_generate = model.generate

    def recording_generate(*args, **kwargs):
        for key in ('input_ids', 'attention_mask', 'pixel_values'):
            value = kwargs.get(key)
            captured[key] = list(value.shape) if hasattr(value, 'shape') else None
        return original_generate(*args, **kwargs)
    model.generate = recording_generate
    try:
        with torch.inference_mode():
            response = model.chat(tokenizer, pixels, prompt, config)
    finally:
        model.generate = original_generate
    if isinstance(response, tuple):
        response = response[0]
    return (str(response), {'prompt_sha256': sha256_bytes(prompt.encode()), 'pixel_values_shape': shape, 'generation_input_shapes': captured, 'input_tensor_keys': sorted(captured)})
