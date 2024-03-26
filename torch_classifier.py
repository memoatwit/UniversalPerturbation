from torch import clip
import torch

def get_clip_model(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device, jit=False)

    return model, preprocess

if __name__ == "__main__":
    model, preprocess =  get_clip_model("RN50")
    """
        text = clip.tokenize(text_list)
        image = preprocess(Image.open(image_path))
        logits_per_image, logits_per_text = model(image, text) #gradients only need this
        image_probs = logits_per_image.softmax(dim=-1)
    """
