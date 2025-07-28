pip install torch transformers diffusers gradio

import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

# Load the pretrained Stable Diffusion model (first run will download weights)
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

# Gradio web interface
interface = gr.Interface(
    fn=generate_image,
    inputs="text",
    outputs="image",
    title="Text to Image Generator",
    description="Enter a prompt and watch the AI generate an image!"
)

if __name__ == "__main__":
    interface.launch()
