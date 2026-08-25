"""One-time: extract HF CLIP image embeddings for NoCaps (MeaCap InvLM memory retrieval).

Requires ViECap-layout images under ./annotations/nocaps/{in_domain,near_domain,out_domain}/.
Output: ./annotations/nocaps/nocaps_corpus_hf_clip.pickle
  rows: [image_id, split, hf_image_embed, captions]  (same order as nocaps_corpus.json)

Usage:
  python scripts/extract_nocaps_hf_clip_embeddings.py \\
    --vl_model /path/to/clip-vit-base-patch32 --local_files_only
"""

import argparse
import json
import os
import pickle

import torch
from tqdm import tqdm

from models.clip_utils import CLIP


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--vl_model', default='openai/clip-vit-base-patch32')
    parser.add_argument('--local_files_only', action='store_true')
    parser.add_argument('--corpus_json', default='./annotations/nocaps/nocaps_corpus.json')
    parser.add_argument('--image_folder', default='./annotations/nocaps/')
    parser.add_argument('--outpath', default='./annotations/nocaps/nocaps_corpus_hf_clip.pickle')
    args = parser.parse_args()

    device = torch.device(args.device)
    vl = CLIP(args.vl_model, local_files_only=args.local_files_only).to(device)

    with open(args.corpus_json, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    results = []
    root = args.image_folder.rstrip('/\\') + '/'
    for ann in tqdm(annotations, desc='HF CLIP nocaps'):
        split = ann['split']
        image_id = ann['image_id']
        caption = ann['caption']
        image_path = root + split + '/' + image_id
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f'Missing image: {image_path}')
        embed = vl.compute_image_representation_from_image_path(image_path).squeeze(0).cpu()
        results.append([image_id, split, embed, caption])

    os.makedirs(os.path.dirname(args.outpath) or '.', exist_ok=True)
    with open(args.outpath, 'wb') as f:
        pickle.dump(results, f)
    print(f'Wrote {len(results)} rows -> {args.outpath}')


if __name__ == '__main__':
    main()
