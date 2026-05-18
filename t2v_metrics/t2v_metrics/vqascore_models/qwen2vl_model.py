import torch
import numpy as np
from PIL import Image
from typing import List, Union
from .qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from .vqa_model import VQAScoreModel
from ..constants import HF_CACHE_DIR

QWEN2_VL_MODELS = {
    # Qwen2.5_VL:
    'qwen2.5-vl-3b': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-3B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen2.5-VL-3B-Instruct',
            'torch_dtype': torch.bfloat16,
        },
    },
    'qwen2.5-vl-7b': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen2.5-VL-7B-Instruct',
            'torch_dtype': torch.bfloat16,
        },
    },
    'qwen2.5-vl-32b': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-32B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen2.5-VL-32B-Instruct',
            'torch_dtype': torch.bfloat16,
        },
    },
    'qwen2.5-vl-72b': {
        'tokenizer': {
            'path': 'Qwen/Qwen2.5-VL-72B-Instruct',
        },
        'model': {
            'path': 'Qwen/Qwen2.5-VL-72B-Instruct',
            'torch_dtype': torch.bfloat16,
        },
    }
}

class Qwen2VLModel(VQAScoreModel):
    def __init__(self,
                 model_name='qwen2.5-vl-7b',
                 device='cuda',
                 cache_dir=HF_CACHE_DIR,
                 checkpoint=None):
        assert model_name in QWEN2_VL_MODELS, f"Model {model_name} not found in QWEN2_VL_MODELS"
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = QWEN2_VL_MODELS[model_name]
        self.checkpoint = checkpoint if checkpoint else self.model_info['model']['path']
        self.load_model()

    def load_model(self):
        # Switch from model dictionary to checkpoint argument
        # model_path = self.model_info['model']['path']
        print('When loading a qwen model, ensure that your model_name or checkpoint contains "qwen2.5". Otherwise, it will be loaded using the "qwen2" config and architecture.')
        model_path = self.checkpoint
        if 'qwen2.5' in model_path or 'qwen2.5' in self.model_name:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                dtype=self.model_info['model']['torch_dtype'],
                cache_dir=self.cache_dir,
            ).to(self.device)

        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                dtype=self.model_info['model']['torch_dtype'],
                cache_dir=self.cache_dir,
            ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            self.model_info['tokenizer']['path'],
            cache_dir=self.cache_dir,
        )
        # Set left-padding for correct batched generation with decoder-only models
        self.processor.tokenizer.padding_side = 'left'
        self.model.eval()

    def load_images(self, images: List[Union[str, Image.Image]]) -> List[Union[torch.Tensor, List[torch.Tensor]]]:
        processed_data = []
        for curr_image in images:
            if isinstance(curr_image, Image.Image):
                processed_data.append({"type": "image", "image": curr_image})
            elif curr_image.lower().endswith('.npy'):  # NumPy file
                np_array = np.load(curr_image)
                if np_array.ndim == 3:  # Single image
                    image = Image.fromarray(np_array.astype('uint8'), 'RGB')
                    processed_data.append({"type": "image", "image": image})
                else:
                    raise ValueError(f"Unexpected shape for NumPy array in {curr_image}")
            else:  # Regular image file
                image = Image.open(curr_image).convert('RGB')
                processed_data.append({"type": "image", "image": image})
        return processed_data

    def forward_sequential(self,
                           paths: List[str],
                           texts: List[str],
                           question_template: str = "Does this image show \"{}\"?",
                           answer_template: str = "Yes") -> torch.Tensor:
        """Sequential (one-by-one) processing. Kept for backward compatibility."""
        assert len(paths) == len(texts), "Number of paths and texts must match"

        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        processed_data = self.load_images(paths)

        lm_probs = []
        for data, question, answer in zip(processed_data, questions, answers):
            messages = [{"role": "user", "content": [data, {"type": "text", "text": question}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True
                )

            scores = outputs.scores[0]
            probs = torch.nn.functional.softmax(scores, dim=-1)
            ans_token_id = self.processor.tokenizer.encode(answer)[0]
            lm_prob = probs[0, ans_token_id].item()
            lm_probs.append(lm_prob)

        return torch.tensor(lm_probs)

    def forward_batched(self,
                        paths: List[str],
                        texts: List[str],
                        question_template: str = "Does this image show \"{}\"?",
                        answer_template: str = "Yes") -> torch.Tensor:
        """Batched processing - single forward pass for all samples (~5x faster)."""
        assert len(paths) == len(texts), "Number of paths and texts must match"

        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        processed_data = self.load_images(paths)

        # Prepare all messages and collect processed texts/images for batched processing
        all_texts = []
        all_images = []

        for data, question in zip(processed_data, questions):
            messages = [{"role": "user", "content": [data, {"type": "text", "text": question}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            all_texts.append(text)
            all_images.extend(image_inputs)

        # Batch process all inputs at once
        inputs = self.processor(
            text=all_texts,
            images=all_images,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        # Single batched generate call
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True
            )

        # Extract probabilities for each sample in the batch
        scores = outputs.scores[0]  # [batch_size, vocab_size]
        probs = torch.nn.functional.softmax(scores, dim=-1)
        ans_token_id = self.processor.tokenizer.encode(answers[0])[0]
        lm_probs = probs[:, ans_token_id]

        return lm_probs.cpu()

    def forward(self,
                paths: List[str],
                texts: List[str],
                question_template: str = "Does this image show \"{}\"?",
                answer_template: str = "Yes") -> torch.Tensor:
        """Forward pass using batched processing by default."""
        return self.forward_batched(paths, texts, question_template, answer_template)
    
    def generate(self,
                images: List[str],
                texts: List[str],
                max_new_tokens: int = 256) -> List[str]:
        assert len(images) == len(texts), "Number of paths and texts must match"
        
        processed_data = self.load_images(images)
        
        generated_texts = []
        for data, text in zip(processed_data, texts):
            messages = [{"role": "user", "content": [data, {"type": "text", "text": text}]}]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()
                generated_texts.append(text)
                
        return generated_texts
