import torch
from PIL import Image
import torchvision.transforms as transforms

# 读取图像并转换为tensor格式
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')  # 读取图像并转换为RGB格式
    transform = transforms.ToTensor()  # 定义转换为tensor的操作
    image_tensor = transform(image)  # 转换图像为tensor格式
    return image_tensor




def RGB2YCbCr(img):
    img = img*255.0
    r,g,b = torch.split(img,1,dim=0)
    y = torch.zeros_like(r)
    cb = torch.zeros_like(r)
    cr = torch.zeros_like(r)
    
    y = 0.257*r+0.504*g+0.098*b+16
    y = y/255.0
    
    cb = -0.148*r-0.291*g+0.439*b+128
    cb = cb/255.0
    
    cr = 0.439*r-0.368*g-0.071*b+128
    cr = cr/255.0
    
    img = torch.cat([y,cb,cr],dim=0)
    return img

img = load_image("/root/autodl-tmp/Data/Flare7Kpp/test_data/real/gt/gt_000000.png")
img = RGB2YCbCr(img)
print(img[0])