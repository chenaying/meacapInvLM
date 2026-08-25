#!/usr/bin/env python3
"""
Organize official NoCaps val images into ViECap layout for MeaCap InvLM / ViECap.

Target layout (must match nocaps_corpus.json + ViT-B32.pickle):
  annotations/nocaps/in_domain/0.jpg
  annotations/nocaps/near_domain/....jpg
  annotations/nocaps/out_domain/....jpg

Inputs:
  1) nocaps_corpus.json  (ViECap list: split, image_id, caption)
  2) val/ folder with OpenImages file names, e.g. 0013ea2087020901.jpg
  3) COCO-style meta JSON with "images": [{id, file_name, open_images_id}, ...]
     (from nocaps.org download, save as nocaps_val_image_info.json)

Usage:
  python scripts/prepare_nocaps_viecap_images.py \\
    --val_dir ./annotations/nocaps/val \\
    --meta_json ./annotations/nocaps/nocaps_val_image_info.json

Use --symlink to save disk space (default: copy).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, Optional


def load_id_to_filename(meta_json: str) -> Dict[str, str]:
    with open(meta_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    mapping: Dict[str, str] = {}
    if not isinstance(data, dict) or 'images' not in data:
        raise ValueError(f'{meta_json} must be COCO-style JSON with an "images" array')

    for img in data['images']:
        file_name = img.get('file_name')
        if not file_name and img.get('open_images_id'):
            oid = str(img['open_images_id'])
            file_name = oid if oid.lower().endswith('.jpg') else f'{oid}.jpg'
        if not file_name:
            continue
        if 'id' in img:
            iid = img['id']
            mapping[str(iid)] = file_name
            mapping[f'{iid}.jpg'] = file_name
        if img.get('open_images_id'):
            oid = str(img['open_images_id'])
            mapping[oid] = file_name
            mapping[f'{oid}.jpg'] = file_name
    return mapping


def resolve_source_name(image_id: str, id_to_file: Dict[str, str]) -> Optional[str]:
    image_id = str(image_id).strip()
    stem, ext = os.path.splitext(image_id)
    if ext.lower() not in ('.jpg', '.jpeg', '.png', '.webp', ''):
        stem, ext = image_id, ''

    # ViECap id like "0.jpg" -> nocaps image id 0
    if stem in id_to_file:
        return id_to_file[stem]
    if image_id in id_to_file:
        return id_to_file[image_id]

    # image_id may already be OpenImages file name
    if ext:
        return image_id
    return f'{image_id}.jpg'


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare ViECap-style NoCaps image folders')
    parser.add_argument('--corpus_json', default='./annotations/nocaps/nocaps_corpus.json')
    parser.add_argument('--val_dir', default='./annotations/nocaps/val',
                        help='Directory of official val images (OpenImages file names)')
    parser.add_argument('--meta_json', default='./annotations/nocaps/nocaps_val_image_info.json',
                        help='COCO-style nocaps val json with "images" field')
    parser.add_argument('--out_dir', default='./annotations/nocaps')
    parser.add_argument('--symlink', action='store_true', help='Symlink instead of copy')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    if not os.path.isfile(args.corpus_json):
        raise FileNotFoundError(f'Missing {args.corpus_json}')
    if not os.path.isdir(args.val_dir):
        raise FileNotFoundError(f'Missing val image dir: {args.val_dir}')
    if not os.path.isfile(args.meta_json):
        raise FileNotFoundError(
            f'Missing {args.meta_json}\n'
            'Download nocaps val image info from https://nocaps.org/download '
            '(validation captions / image info JSON) and save as nocaps_val_image_info.json'
        )

    with open(args.corpus_json, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    id_to_file = load_id_to_filename(args.meta_json)
    val_index = {}
    for fname in os.listdir(args.val_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            val_index[fname] = os.path.join(args.val_dir, fname)

    ok, miss_src, miss_map = 0, 0, 0
    for ann in corpus:
        split = ann['split']
        image_id = ann['image_id']
        src_name = resolve_source_name(image_id, id_to_file)
        if src_name is None:
            miss_map += 1
            continue
        src_path = val_index.get(src_name) or os.path.join(args.val_dir, src_name)
        if not os.path.isfile(src_path):
            miss_src += 1
            continue

        dst_dir = os.path.join(args.out_dir, split)
        dst_path = os.path.join(dst_dir, image_id if '.' in str(image_id) else f'{image_id}.jpg')
        if args.dry_run:
            print(f'[dry-run] {src_path} -> {dst_path}')
            ok += 1
            continue

        os.makedirs(dst_dir, exist_ok=True)
        if os.path.exists(dst_path):
            ok += 1
            continue
        if args.symlink:
            os.symlink(os.path.abspath(src_path), dst_path)
        else:
            shutil.copy2(src_path, dst_path)
        ok += 1

    print(f'Done: {ok}/{len(corpus)} images -> {args.out_dir}/{{in_domain,near_domain,out_domain}}/')
    if miss_map:
        print(f'  skipped (no id mapping): {miss_map}')
    if miss_src:
        print(f'  skipped (file not in val_dir): {miss_src}')
    sample = os.path.join(args.out_dir, 'in_domain', '0.jpg')
    if os.path.isfile(sample):
        print(f'  OK sample: {sample}')
    else:
        print(f'  WARN: sample missing: {sample} (check corpus first image_id)')


if __name__ == '__main__':
    main()
