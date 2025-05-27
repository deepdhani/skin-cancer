from pathlib import Path
from PIL import Image

# Define your dummy dataset folder path here
dummy_path = Path('/Users/deepak/Desktop/pbl/pbl2/dummy_data')

# List of required classes (folder names)
required_classes = [
    'Actinic keratoses',
    'Basal cell carcinoma',
    'Benign keratosis',
    'Dermatofibroma',
    'Melanocytic nevi',
    'Melanoma',
    'Vascular lesions'
]

def create_dummy_image(path: Path):
    # Create a 1x1 pixel black PNG image
    img = Image.new('RGB', (1, 1), color=(0, 0, 0))
    img.save(path)

def fix_dummy_dataset(base_path: Path, classes: list):
    base_path.mkdir(parents=True, exist_ok=True)

    for cls in classes:
        class_folder = base_path / cls
        if not class_folder.exists():
            print(f"Creating missing folder: {class_folder}")
            class_folder.mkdir(parents=True)

            # Add a dummy image so fastai can load the folder
            dummy_image_path = class_folder / 'dummy.png'
            create_dummy_image(dummy_image_path)
            print(f"Added dummy image: {dummy_image_path}")
        else:
            # Check if folder has at least one image file
            image_files = list(class_folder.glob('*.*'))
            if not image_files:
                print(f"Folder '{cls}' is empty, adding dummy image.")
                dummy_image_path = class_folder / 'dummy.png'
                create_dummy_image(dummy_image_path)
                print(f"Added dummy image: {dummy_image_path}")
            else:
                print(f"Folder '{cls}' already exists with images.")

if __name__ == "__main__":
    fix_dummy_dataset(dummy_path, required_classes)
    print("Dummy dataset folders are fixed!")