"""
DPG-Bench utilities for loading metadata.
"""

import json
import os
from typing import Dict, List

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class MPLUG:
    """
    mPLUG VQA model wrapper with batch processing support.

    Example:
        >>> from dpg_evaluator.utils import MPLUG, load_prompt2id, load_dpg_metadata
        >>> from dpg_evaluator.evaluator import evaluate
        >>> from PIL import Image
        >>>
        >>> # Load model (auto-downloads from HuggingFace if not found)
        >>> model = MPLUG()  # or MPLUG(model_dir='path/to/checkpoint')
        >>>
        >>> # Load metadata
        >>> prompt2id = load_prompt2id()
        >>> question_dict = load_dpg_metadata()
        >>>
        >>> # Prepare data
        >>> images = [Image.open('0.png'), Image.open('1.png')]
        >>> prompts = ['A cat sitting on a mat.', 'A dog running in the park.']
        >>>
        >>> # Evaluate
        >>> mean_score, l1_scores, l2_scores = evaluate(
        ...     images, prompts, prompt2id, question_dict, model.batch_vqa
        ... )
    """

    REPO_ID = 'rae-t2i/mplug_visual-question-answering_coco_large_en'

    def __init__(self, model_dir: str = None, device: str = 'cuda'):
        from .mplug import MPlug
        from .mplug.constants import Tasks

        if model_dir is None or not os.path.exists(model_dir):
            from huggingface_hub import snapshot_download
            model_dir = snapshot_download(repo_id=self.REPO_ID)
            print(f'Using mPLUG model from: {model_dir}')

        self.device = torch.device(device)
        self.model = MPlug.from_pretrained(
            model_dir, task=Tasks.visual_question_answering
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.tokenizer = self.model.tokenizer
        self.image_res = self.model.config.image_res

        self.transform = transforms.Compose([
            transforms.Resize(
                (self.image_res, self.image_res),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
        ])

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return self.transform(image)

    def preprocess_images(self, images: List[Image.Image]) -> torch.Tensor:
        tensors = [self.preprocess_image(img) for img in images]
        return torch.stack(tensors).to(self.device)

    def preprocess_questions(self, questions: List[str]):
        return self.tokenizer(
            questions,
            padding='longest',
            truncation=True,
            max_length=35,
            return_tensors='pt'
        ).to(self.device)

    @torch.no_grad()
    def vqa(self, image: Image.Image, question: str) -> str:
        """Single image-question pair inference."""
        return self.batch_vqa([image], [question])[0]

    @torch.no_grad()
    def batch_vqa(self, images: List[Image.Image], questions: List[str], use_fp16: bool = True) -> List[str]:
        """Batch inference for multiple image-question pairs."""
        image_tensor = self.preprocess_images(images)
        question_input = self.preprocess_questions(questions)

        with torch.amp.autocast('cuda', enabled=use_fp16):
            topk_ids, _ = self.model(
                image=image_tensor,
                question=question_input,
                train=False
            )

        answers = []
        for ids in topk_ids:
            answer = self.tokenizer.decode(ids[0]).replace('[SEP]', '').replace('[CLS]', '').replace('[PAD]', '').strip()
            answers.append(answer)

        return answers


def load_prompt2id(
    json_path: str = os.path.join(os.path.dirname(__file__), 'prompt2id.json')
) -> Dict[str, str]:
    """
    Load prompt to item_id mapping.
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def load_dpg_metadata(
    csv_path: str = os.path.join(os.path.dirname(__file__), 'dpg_bench.csv')
) -> Dict[str, Dict]:
    """
    Load and parse DPG-Bench metadata from CSV.
    Uses item_id as the key.
    """
    question_dict = {}
    previous_id = ''

    data = pd.read_csv(csv_path)
    for i, line in data.iterrows():
        if i == 0:  # skip first row like reference
            continue
        current_id = str(line.item_id)
        qid = int(line.proposition_id)

        dependency_list = [
            int(d.strip()) for d in str(line.dependency).split(',')
        ]

        if current_id == previous_id:
            question_dict[current_id]['qid2tuple'][qid] = line.tuple
            question_dict[current_id]['qid2dependency'][qid] = dependency_list
            question_dict[current_id]['qid2question'][qid] = line.question_natural_language
        else:
            question_dict[current_id] = {
                'qid2tuple': {qid: line.tuple},
                'qid2dependency': {qid: dependency_list},
                'qid2question': {qid: line.question_natural_language}
            }

        previous_id = current_id

    return question_dict
