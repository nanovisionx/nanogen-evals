"""
Constants for mPLUG model (replacing modelscope.utils.constant).
"""


class Tasks:
    visual_question_answering = 'visual-question-answering'
    image_captioning = 'image-captioning'
    image_text_retrieval = 'image-text-retrieval'
    video_question_answering = 'video-question-answering'


class ModelFile:
    VOCAB_FILE = 'vocab.txt'
    TORCH_MODEL_BIN_FILE = 'pytorch_model.bin'
