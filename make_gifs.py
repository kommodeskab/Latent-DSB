from src.lightning_modules import DSB
from src.networks import MediumUNet, StableDiffusionXL
from src.utils import get_ckpt_path
import torch
from PIL import Image, ExifTags
from torch import Tensor
import numpy as np
import os

new_path_name = "søren_2"
seed = 0


torch.manual_seed(0)
os.makedirs(f"gifs/{new_path_name}", exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
img_path = f'people_images/{new_path_name}.jfif'

forward_model = MediumUNet()
backward_model = MediumUNet()
encoder = StableDiffusionXL()
ckpt_path = get_ckpt_path(experiment_id='230325104152')
dsb = DSB.load_from_checkpoint(ckpt_path, forward_model=forward_model, backward_model=backward_model, encoder_decoder=encoder)
dsb = dsb.to(device)
dsb : DSB

def load_image_with_orientation(image_path):
    image = Image.open(image_path)
    
    # Apply EXIF orientation if it exists
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation, None)
            if orientation_value == 3:
                image = image.rotate(180, expand=True)
            elif orientation_value == 6:
                image = image.rotate(270, expand=True)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # If no EXIF data or orientation tag is found, do nothing
        pass
    
    return image

def image_to_tensor(image : Image.Image) -> Tensor:
    image = image.resize((128, 128))
    image = np.array(image)
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image = image * 2 - 1
    return image

img = load_image_with_orientation(img_path)
img_tensor = image_to_tensor(img).repeat(16, 1, 1, 1)
print("img_tensor", img_tensor.shape)

img_tensor = img_tensor.to(device)
with torch.no_grad():
    img_encoded = dsb.encode(img_tensor)
    img_encoded_translated = dsb.sample(img_encoded, True, False, True, 'inference')
    # img_encoded_translated = torch.randn_like(img_encoded)

weights = torch.linspace(0, 1, 100).view(-1, 1, 1, 1, 1).to(device)
img_encoded_interpolation = img_encoded * (1 - weights) + img_encoded_translated * weights

for i in range(16):
    img_decoded_interpolation = dsb.decode(img_encoded_interpolation[:, i])
    img_decoded_interpolation = img_decoded_interpolation.detach().cpu().clamp(-1, 1).permute(0, 2, 3, 1).numpy()
    img_decoded_interpolation = (img_decoded_interpolation + 1) / 2

    # Create a GIF. img_decoded.interpolation have shape [100, 128, 128, 3]

    def create_gif(images, filename='interpolation.gif', duration=100):
        images = [Image.fromarray((img * 255).astype(np.uint8)) for img in images]
        images[0].save(filename, save_all=True, append_images=images[1:], duration=duration, loop=1)
        print(f"GIF saved as {filename}")
        
    create_gif(img_decoded_interpolation, filename=f'gifs/{new_path_name}/interpolation_{i}.gif', duration=100)