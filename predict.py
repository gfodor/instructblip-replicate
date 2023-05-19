import os

os.environ["HUGGINGFACE_HUB_CACHE"] = "cache/huggingface/hub"
os.environ["TORCH_HOME"] = "cache/torch"

from cog import BasePredictor, Input, Path
from lavis.models import load_model_and_preprocess
from PIL import Image

model_name = "blip2_t5_instruct"
model_type = "flant5xxl"
device = "cuda"

# Predict

class Predictor(BasePredictor):
    def setup(self):
        model, vis_processors, _ = load_model_and_preprocess(
            name=model_name,
            model_type=model_type,
            is_eval=True,
            device=device
        )

        self.model = model
        self.vis_processors = vis_processors

    def infer(self, image, prompt, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, use_nucleus_sampling):
        image = self.vis_processors["eval"](image).unsqueeze(0).to(device)

        samples = {
            "image": image,
            "prompt": prompt,
        }

        output = self.model.generate(
            samples,
            length_penalty=float(len_penalty),
            repetition_penalty=float(repetition_penalty),
            num_beams=beam_size,
            max_length=max_len,
            min_length=min_len,
            top_p=top_p,
            use_nucleus_sampling=use_nucleus_sampling
        )

        return output[0]

    def predict(
        self,
        image_path: Path = Input(description="image"),
        prompt: str = Input(description="prompt", default="describe the image"),
        min_len: int = Input(description="min length", default=1),
        max_len: int = Input(description="max length", default=200),
        beam_size: int = Input(description="beam size", default=5),
        len_penalty: float = Input(description="length penalty", default=1),
        repetition_penalty: float = Input(description="repetition penalty", default=3),
        top_p: float = Input(description="top p", default=0.9),
        use_nucleus_sampling: bool = Input(description="use nucleus sampling", default=False)
    ) -> str:
        """Get caption"""
        image = Image.open(image_path).convert("RGB")
        return self.infer(image, prompt, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, use_nucleus_sampling)
