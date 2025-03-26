from src.lightning_modules import DSB
from src.networks import MediumUNet, StableDiffusionXL
from src.utils import get_ckpt_path
import torch
from PIL import Image, ExifTags
import numpy as np
from torch import Tensor
import io

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
folder_path = 'people_images'
file_name = 'andreas_billede.jpg'
img_path = f'{folder_path}/{file_name}'
to_woman = True

def load_model(device : torch.device) -> DSB:
    print("Loading models...")
    forward_model = MediumUNet()
    backward_model = MediumUNet()
    print("Loading encoder..")
    encoder = StableDiffusionXL()
    print("Getting checkpoint path..")
    ckpt_path = get_ckpt_path(experiment_id='230325104152')
    print("Loading DSB from checkpoint..")
    dsb = DSB.load_from_checkpoint(ckpt_path, forward_model=forward_model, backward_model=backward_model, encoder_decoder=encoder)
    dsb = dsb.to(device)
    return dsb

def image_to_tensor(image : Image.Image) -> Tensor:
    image = image.resize((128, 128))
    image = np.array(image)
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image = image * 2 - 1
    return image

def tensor_to_image(tensor : Tensor) -> Image.Image:
    tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    tensor = (tensor + 1) / 2
    tensor = (tensor * 255).astype('uint8')
    return Image.fromarray(tensor)

def create_gif(frames : list[Tensor]) -> io.BytesIO:
    frames : list[Image.Image] = [tensor_to_image(f) for f in frames]
    gif_buffer = io.BytesIO()
    frames[0].save(
        gif_buffer,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=100,
        loop=1,
    )
    gif_buffer.seek(0)
    return gif_buffer

dsb = load_model(device)
dsb = dsb.to(device)

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

img = load_image_with_orientation(img_path)
img = image_to_tensor(img)
img = img.to(device)

def gif_generator(dsb : DSB, img_tensor : Tensor):
    with torch.no_grad():
        img_encoded = dsb.encode(img_tensor)
        encoded_trajectory = dsb.sample(img_encoded, forward=to_woman, return_trajectory=True, show_progress=True, noise='inference')
        encoded_trajectory = encoded_trajectory.flatten(0, 1)
        decoded_trajectory = dsb.decode(encoded_trajectory)
    gif_buffer = create_gif(decoded_trajectory)
    return gif_buffer

gif_buffer = gif_generator(dsb, img)
# save the gif to a file
with open(f'{folder_path}/gif_{file_name}.gif', 'wb') as f:
    f.write(gif_buffer.getbuffer())