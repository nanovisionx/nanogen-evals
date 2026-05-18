"""
Make sure you do pip install -e . before running this test.
python test/test_vqascore.py --model clip-flant5-xl
python test/test_vqascore.py --model clip-flant5-xxl
python test/test_vqascore.py --model qwen2.5-vl-3b
python test/test_vqascore.py --model qwen2.5-vl-7b
python test/test_vqascore.py --model qwen3-vl-2b
python test/test_vqascore.py --model qwen3-vl-4b
python test/test_vqascore.py --model qwen3-vl-8b
python test/test_vqascore.py --model qwen3.5-4b
python test/test_vqascore.py --model qwen3.5-9b
python test/test_vqascore.py --model qwen3.5-27b
"""
import argparse
import os
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from t2v_metrics import VQAScore


def test_vqascore(args):
    """Test VQAScore with a sample image and prompt."""
    # Initialize VQAScore model
    model = VQAScore(model=args.model, device='cuda')

    # Define image path and prompt
    paths = ["sample0.png", "sample1.png"]
    prompts = ["three tomatoes (above) and two cucumbers (below) from top view", "a corgi playing in the garden"]
    for img_path, prompt in zip(paths, prompts):
        image = Image.open(os.path.join('assets', img_path))
        score = model(images=[image], texts=[prompt])[0]
        print(f"Image: {img_path}")
        print(f"Prompt: {prompt}")
        print(f"VQAScore: {score.item():.4f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='clip-flant5-xl')
    args = parser.parse_args()
    test_vqascore(args)
