"""Shared MeaCap InvLM prompt building for viecap_inference.py and validation.py."""

import copy
import os
import json
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

from models.clip_utils import CLIP
from meacap_utils.detect_utils import retrieve_concepts
from utils import compose_discrete_prompts

cpu_device = torch.device('cpu')


def hf_load_kwargs(local_files_only: bool) -> dict:
    return {'local_files_only': True} if local_files_only else {}


def load_memory_clip(model_path: str, local_files_only: bool) -> CLIP:
    import inspect

    if local_files_only:
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

    params = inspect.signature(CLIP.__init__).parameters
    if 'local_files_only' in params:
        return CLIP(model_path, local_files_only=local_files_only)
    return CLIP(model_path)


class MeaCapInvLMResources:
    """Memory bank + retrieval models (load once for batch evaluation)."""

    def __init__(self, args, device: torch.device):
        hf_kw = hf_load_kwargs(getattr(args, 'local_files_only', False))

        self.device = device
        self.memory_caption_num = args.memory_caption_num

        self.vl_model = load_memory_clip(args.vl_model, getattr(args, 'local_files_only', False))
        self.vl_model = self.vl_model.to(device)

        self.wte_model = SentenceTransformer(args.wte_model_path)
        self.parser_tokenizer = AutoTokenizer.from_pretrained(args.parser_checkpoint, **hf_kw)
        self.parser_model = AutoModelForSeq2SeqLM.from_pretrained(args.parser_checkpoint, **hf_kw)
        self.parser_model.eval()
        self.parser_model.to(device)

        memory_id = args.memory_id
        memory_dir = os.path.join('data/memory', memory_id)
        with open(os.path.join(memory_dir, 'memory_captions.json'), 'r', encoding='utf-8') as f:
            self.memory_captions = json.load(f)
        self.memory_clip_embeddings = torch.load(
            os.path.join(memory_dir, 'memory_clip_embeddings.pt'),
            map_location=device,
        )

        if memory_id in ('cc3m', 'ss1m'):
            self.retrieve_on_cpu = True
            self.vl_model_retrieve = copy.deepcopy(self.vl_model).to(cpu_device)
            self.memory_clip_embeddings = self.memory_clip_embeddings.to(cpu_device)
        else:
            self.retrieve_on_cpu = False
            self.vl_model_retrieve = self.vl_model

        print(f'[MeaCap InvLM] memory bank "{memory_id}" loaded ({len(self.memory_captions)} captions).')


def _memory_retrieval_from_embeds(
    resources: MeaCapInvLMResources,
    batch_image_embeds: torch.Tensor,
) -> List[str]:
    device = resources.device
    batch_image_embeds = batch_image_embeds.to(device)
    if batch_image_embeds.dim() == 1:
        batch_image_embeds = batch_image_embeds.unsqueeze(0)
    batch_image_embeds = batch_image_embeds / batch_image_embeds.norm(dim=-1, keepdim=True)

    if not resources.retrieve_on_cpu:
        clip_score, _ = resources.vl_model_retrieve.compute_image_text_similarity_via_embeddings(
            batch_image_embeds, resources.memory_clip_embeddings
        )
    else:
        batch_cpu = batch_image_embeds.to(cpu_device)
        clip_score, _ = resources.vl_model_retrieve.compute_image_text_similarity_via_embeddings(
            batch_cpu, resources.memory_clip_embeddings
        )
        clip_score = clip_score.to(device)

    select_ids = clip_score.topk(resources.memory_caption_num, dim=-1)[1].squeeze(0)
    select_captions = [resources.memory_captions[i] for i in select_ids]

    return retrieve_concepts(
        parser_model=resources.parser_model,
        parser_tokenizer=resources.parser_tokenizer,
        wte_model=resources.wte_model,
        select_memory_captions=select_captions,
        image_embeds=batch_image_embeds,
        device=device,
    )


def retrieve_memory_concepts_from_embeds(
    resources: MeaCapInvLMResources,
    batch_image_embeds: torch.Tensor,
) -> List[str]:
    """Memory retrieval using precomputed HF CLIP image embeddings (no jpg on disk)."""
    return _memory_retrieval_from_embeds(resources, batch_image_embeds)


def retrieve_memory_concepts(resources: MeaCapInvLMResources, image_path: str) -> List[str]:
    device = resources.device
    batch_image_embeds = resources.vl_model.compute_image_representation_from_image_path(image_path)
    return _memory_retrieval_from_embeds(resources, batch_image_embeds)


def _warn_openai_pickle_memory_once(args) -> None:
    if getattr(args, '_warned_openai_memory', False):
        return
    print(
        '[warn] --invlm_memory_from_openai_pickle: ViECap pickle uses OpenAI CLIP, '
        'but memory bank uses HF CLIP. Retrieval is approximate, not MeaCap-paper faithful.'
    )
    args._warned_openai_memory = True


def retrieve_memory_concepts_for_validation(
    args,
    resources: MeaCapInvLMResources,
    image_features: torch.Tensor,
    image_path: str,
    image_id: str = None,
    split: str = None,
) -> List[str]:
    """Pick memory retrieval source: disk image, HF embed pickle, or OpenAI pickle (fallback)."""
    if getattr(args, 'nocaps_hf_clip_pickle', None) and getattr(args, '_nocaps_hf_embed_lookup', None):
        lookup = args._nocaps_hf_embed_lookup
        key = (image_id, split) if split is not None else image_id
        hf_embed = lookup.get(key) or lookup.get(image_id)
        if hf_embed is None:
            raise KeyError(f'No HF CLIP embedding in pickle for image_id={image_id}, split={split}')
        return retrieve_memory_concepts_from_embeds(resources, hf_embed.unsqueeze(0).float())

    if getattr(args, 'invlm_memory_from_openai_pickle', False):
        _warn_openai_pickle_memory_once(args)
        return retrieve_memory_concepts_from_embeds(resources, image_features)

    if image_path is None:
        raise ValueError('InvLM memory retrieval needs image_path, --nocaps_hf_clip_pickle, or --invlm_memory_from_openai_pickle')
    return retrieve_memory_concepts(resources, image_path)


def load_nocaps_hf_clip_lookup(pickle_path: str) -> dict:
    import pickle as pkl

    with open(pickle_path, 'rb') as f:
        rows = pkl.load(f)
    lookup = {}
    for row in rows:
        image_id, split, hf_embed, _captions = row[0], row[1], row[2], row[3]
        lookup[(image_id, split)] = hf_embed
        lookup[image_id] = hf_embed
    return lookup


def invlm_skips_disk_images(args) -> bool:
    return bool(
        getattr(args, 'invlm_memory_from_openai_pickle', False)
        or getattr(args, 'nocaps_hf_clip_pickle', None)
    )


def build_embeddings(
    args,
    model,
    tokenizer,
    continuous_embeddings: torch.Tensor,
    detected_objects: List[str],
    device: torch.device,
) -> torch.Tensor:
    discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects).unsqueeze(dim=0).to(device)
    discrete_embeddings = model.word_embed(discrete_tokens)

    if args.only_hard_prompt:
        return discrete_embeddings
    if args.soft_prompt_first:
        return torch.cat((continuous_embeddings, discrete_embeddings), dim=1)
    return torch.cat((discrete_embeddings, continuous_embeddings), dim=1)


NOCAPS_SPLIT_DIR_ALIASES = {
    'in_domain': ('in_domain', 'in-domain', 'In-Domain', 'indomain'),
    'near_domain': ('near_domain', 'near-domain', 'Near-Domain', 'neardomain'),
    'out_domain': ('out_domain', 'out-domain', 'Out-Domain', 'outdomain'),
}


def load_nocaps_image_id_map(image_folder: str, corpus_json_path: str = None) -> dict:
    """Map ViECap pickle ids (e.g. 0.jpg) to official NoCaps file_name (e.g. 0013ea....jpg)."""
    mapping = {}

    def _add(key, file_name: str) -> None:
        if not file_name:
            return
        if not os.path.splitext(file_name)[1]:
            file_name = f'{file_name}.jpg'
        key_s = str(key).strip()
        mapping[key_s] = file_name
        stem, _ = os.path.splitext(key_s)
        mapping[stem] = file_name
        mapping[f'{stem}.jpg'] = file_name

    meta_paths = []
    if corpus_json_path:
        meta_paths.append(corpus_json_path)
    meta_paths.extend([
        os.path.join(image_folder, 'nocaps_val_image_info.json'),
        os.path.join(image_folder, 'image_info.json'),
        os.path.join(image_folder, 'nocaps_val_4500_captions.json'),
    ])

    for path in meta_paths:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if isinstance(data, dict) and 'images' in data:
            for img in data['images']:
                file_name = img.get('file_name')
                if not file_name and img.get('open_images_id'):
                    oid = str(img['open_images_id'])
                    file_name = oid if oid.endswith('.jpg') else f'{oid}.jpg'
                if not file_name:
                    continue
                if 'id' in img:
                    _add(img['id'], file_name)
                if img.get('open_images_id'):
                    _add(img['open_images_id'], file_name)
            if mapping:
                print(f'[nocaps] image id map: {len(mapping)} keys from {path}')
                return mapping

    return mapping


def ensure_nocaps_image_map(args) -> None:
    if getattr(args, '_nocaps_image_map', None) is None:
        corpus = getattr(args, 'path_of_val_datasets', None)
        meta = getattr(args, 'nocaps_meta_json', None)
        args._nocaps_image_map = load_nocaps_image_id_map(
            args.image_folder,
            meta or corpus,
        )


def _normalize_image_id(image_id) -> str:
    image_id = str(image_id).strip()
    if not os.path.splitext(image_id)[1]:
        image_id = f'{image_id}.jpg'
    return image_id


def resolve_image_path(args, image_id: str, split: str = None) -> str:
    """Build filesystem path for HF CLIP memory retrieval (must exist on disk)."""
    image_id = _normalize_image_id(image_id)
    image_folder = args.image_folder.rstrip('/\\')
    stem = os.path.splitext(image_id)[0]

    candidates = []
    image_map = getattr(args, '_nocaps_image_map', None) or {}
    alt_name = image_map.get(image_id) or image_map.get(stem)
    if alt_name:
        for sub in ('val', 'images', 'validation', ''):
            if sub:
                candidates.append(os.path.join(image_folder, sub, alt_name))
            else:
                candidates.append(os.path.join(image_folder, alt_name))

    if split is not None:
        split_dirs = NOCAPS_SPLIT_DIR_ALIASES.get(split, (split,))
        for split_dir in split_dirs:
            candidates.append(os.path.join(image_folder, split_dir, image_id))
            if alt_name:
                candidates.append(os.path.join(image_folder, split_dir, alt_name))
        candidates.append(os.path.join(image_folder, 'val', image_id))
        if alt_name:
            candidates.append(os.path.join(image_folder, 'val', alt_name))
    else:
        candidates.append(os.path.join(image_folder, image_id))

    # de-duplicate while preserving order
    seen = set()
    unique_candidates = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique_candidates.append(path)
    candidates = unique_candidates

    for path in candidates:
        if os.path.isfile(path):
            return path

    # basename index built once for NoCaps (optional fallback)
    index = getattr(args, '_nocaps_image_index', None)
    if index is not None:
        if image_id in index:
            return index[image_id]
        stem = os.path.splitext(image_id)[0]
        if stem in index:
            return index[stem]

    tried = '\n  '.join(candidates[:6])
    raise FileNotFoundError(
        f'NoCaps image not found for image_id={image_id}, split={split}.\n'
        f'Tried:\n  {tried}\n'
        f'Please provide NoCaps val images:\n'
        f'  (A) ViECap layout: {image_folder}/in_domain/0.jpg (from checkpoints.zip), or\n'
        f'  (B) Official layout: {image_folder}/val/<openimages_id>.jpg plus image_info JSON\n'
        f'      (nocaps_val_image_info.json / COCO-style nocaps json with "images" field).'
    )


def build_nocaps_image_index(image_folder: str) -> dict:
    """Map basename / stem -> absolute image path under image_folder."""
    index = {}
    if not os.path.isdir(image_folder):
        return index
    for dirpath, _, filenames in os.walk(image_folder):
        for fname in filenames:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue
            full = os.path.join(dirpath, fname)
            index[fname] = full
            stem, _ = os.path.splitext(fname)
            index.setdefault(stem, full)
    return index


def check_nocaps_images_available(args, sample_paths: List[tuple]) -> None:
    """Preflight: verify a few NoCaps images exist before 4500-iter loop."""
    if not getattr(args, 'use_meacap_invlm', False):
        return
    if invlm_skips_disk_images(args):
        print('[nocaps] InvLM memory: using precomputed embeddings (no jpg required).')
        return
    ensure_nocaps_image_map(args)
    if not getattr(args, '_nocaps_image_index', None):
        args._nocaps_image_index = build_nocaps_image_index(args.image_folder)
    missing = 0
    for image_id, split in sample_paths[:5]:
        try:
            resolve_image_path(args, image_id, split)
        except FileNotFoundError:
            missing += 1
    if missing == len(sample_paths[:5]):
        raise FileNotFoundError(
            f'No NoCaps images found under {args.image_folder}. '
            'InvLM requires raw images for HF CLIP memory retrieval. '
            'ViECap baseline (--use_meacap_invlm False) can run with pickle features only.'
        )


def hard_prompt_embeddings(
    args,
    model,
    tokenizer,
    continuous_embeddings: torch.Tensor,
    image_features: torch.Tensor,
    image_path: str = None,
    device: torch.device = None,
    invlm_resources: MeaCapInvLMResources = None,
    entities_text: List[str] = None,
    texts_embeddings: torch.Tensor = None,
    image_id: str = None,
    split: str = None,
) -> torch.Tensor:
    """ViECap entity classifier or MeaCap memory concepts."""
    if getattr(args, 'use_meacap_invlm', False):
        detected_objects = retrieve_memory_concepts_for_validation(
            args, invlm_resources, image_features, image_path, image_id, split,
        )
    else:
        from retrieval_categories import image_text_simiarlity, top_k_categories

        logits = image_text_simiarlity(
            texts_embeddings, temperature=args.temperature, images_features=image_features
        )
        detected_objects, _ = top_k_categories(entities_text, logits, args.top_k, args.threshold)
        detected_objects = detected_objects[0]

    return build_embeddings(args, model, tokenizer, continuous_embeddings, detected_objects, device)
