from fastai.vision.all import *
from fastai.vision.augment import Resize
import gradio as gr
from pathlib import Path
import torch

# Constants
NUM_EPOCHS = 5
MODEL_WEIGHTS_FILE = "melanoma_model.pth"
PATH_TO_MODELS_DIR = Path("..") / "models"
PATH_TO_DATA = Path("..") / "dummy_data"  # expects train/ and valid/ folders here

def setup_model():
    device = torch.device("cpu")  # Using CPU due to MPS limitations

    # Load dataset with manual train/valid split
    dls = ImageDataLoaders.from_folder(
        PATH_TO_DATA,
        train="train",
        valid="valid",
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(),
        device=device
    )

    # Create and fine-tune the learner
    learn = vision_learner(dls, resnet18, metrics=accuracy).to_fp32()
    learn.dls.device = device
    learn.fine_tune(NUM_EPOCHS)

    # Save model weights
    learn.path = PATH_TO_MODELS_DIR
    learn.save(MODEL_WEIGHTS_FILE.replace(".pth", ""))
    print(f"✅ Model trained and saved to {PATH_TO_MODELS_DIR / MODEL_WEIGHTS_FILE}")

    return learn

# Load the model
learn = setup_model()

# Define prediction function
def classify_image(img):
    pred, pred_idx, probs = learn.predict(img)
    return dict(zip(learn.dls.vocab, map(float, probs)))

# Set up Gradio interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Skin Cancer Classifier",
    description="Upload a skin lesion image to classify it using a CNN model."
)

iface.launch()
