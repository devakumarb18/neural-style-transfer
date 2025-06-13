import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# --------- Load Image Function ---------
def load_image(img_path, max_size=400):
    image = Image.open(img_path).convert('RGB')
    size = max_size if max(image.size) > max_size else max(image.size)
    
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image

# --------- Image Convert Function (You gave) ---------
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * 0.5 + 0.5  # unnormalize
    image = image.clip(0, 1)
    return image

# --------- Load Content and Style Images ---------
content = load_image("content.jpg")
style = load_image("style.jpg")

# --------- Display the Images ---------
plt.imshow(im_convert(content))
plt.title("Content Image")
plt.axis('off')
plt.show()

plt.imshow(im_convert(style))
plt.title("Style Image")
plt.axis('off')
plt.show()

