import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import glob
import random


from scipy import ndimage
from skimage import morphology
from skimage.measure import label
from skimage.filters import rank
from skimage.morphology import disk
from skimage import color
from skimage.measure import regionprops

import torchvision.transforms.functional as TF
from torch.distributions import Normal
import torch
import numpy as np
import torch

def plot_light_pos(input_img,threshold):
	#input should be a three channel tensor with shape [C,H,W]
	#Out put the position (x,y) in int
	luminance=0.3*input_img[0]+0.59*input_img[1]+0.11*input_img[2]
	luminance_mask=luminance>threshold
	luminance_mask_np=luminance_mask.numpy()
	struc = disk(3)
	img_e = ndimage.binary_erosion(luminance_mask_np, structure = struc)
	img_ed = ndimage.binary_dilation(img_e, structure = struc)

	labels = label(img_ed)
	if labels.max() == 0:
		#print("Light source not found.")
		return (255,255)
	else:
		largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
		largestCC=largestCC.astype(int)
		properties = regionprops(largestCC, largestCC)
		weighted_center_of_mass = properties[0].weighted_centroid
		# print("Light source detected in position: x:",int(weighted_center_of_mass[1]),",y:",int(weighted_center_of_mass[0]))
		return (int(weighted_center_of_mass[1]),int(weighted_center_of_mass[0]))

class RandomGammaCorrection(object):
	def __init__(self, gamma = None):
		self.gamma = gamma
	def __call__(self,image):
		if self.gamma == None:
			# more chances of selecting 0 (original image)
			gammas = [0.5,1,2]
			self.gamma = random.choice(gammas)
			return TF.adjust_gamma(image, self.gamma, gain=1)
		elif isinstance(self.gamma,tuple):
			gamma=random.uniform(*self.gamma)
			return TF.adjust_gamma(image, gamma, gain=1)
		elif self.gamma == 0:
			return image
		else:
			return TF.adjust_gamma(image,self.gamma,gain=1)

class TranslationTransform(object):
    def __init__(self, position):
        self.position = position

    def __call__(self, x):
        return TF.affine(x,angle=0, scale=1,shear=[0,0], translate= list(self.position))

def remove_background(image):
	#the input of the image is PIL.Image form with [H,W,C]
	image=np.float32(np.array(image))
	_EPS=1e-7
	rgb_max=np.max(image,(0,1))
	rgb_min=np.min(image,(0,1))
	image=(image-rgb_min)*rgb_max/(rgb_max-rgb_min+_EPS)
	image=torch.from_numpy(image)
	return image

def generate_flare(img_HR, scattering_flare_img, reflective_flare_img, light_source_img):
	gamma=np.random.uniform(1.8,2.2)
	to_tensor=transforms.ToTensor()
	adjust_gamma=RandomGammaCorrection(gamma)
	adjust_gamma_reverse=RandomGammaCorrection(1/gamma)
	color_jitter=transforms.ColorJitter(brightness=(0.8,3),hue=0.0)
	transform_HR = transforms.Compose([transforms.RandomCrop((512,512),pad_if_needed=True,padding_mode='reflect'),
									transforms.RandomHorizontalFlip(),
									transforms.RandomVerticalFlip()])
	
	img_HR = to_tensor(img_HR)
	img_HR = adjust_gamma(img_HR)
	img_HR = transform_HR(img_HR)
	
	sigma_chi=0.01*np.random.chisquare(df=1)
	img_HR=Normal(img_HR,sigma_chi).sample()
	gain=np.random.uniform(1,1.2)
	flare_DC_offset=np.random.uniform(-0.02,0.02)
	img_HR=gain*img_HR
	img_HR=torch.clamp(img_HR,min=0,max=1)

	light_pos=plot_light_pos(img_HR,0.97**gamma)
	light_pos=[light_pos[0]-256,light_pos[1]-256]
	#traslate=TranslationTransform(light_pos)
	transform_flare=transforms.Compose([transforms.RandomHorizontalFlip(),
							  transforms.RandomVerticalFlip(),
                              transforms.RandomAffine(degrees=(0,360),scale=(0.8,1.5),translate=(0,0),shear=(-20,20)),
							  TranslationTransform(light_pos),
                              transforms.CenterCrop((512,512)),
                              ])
	reflective_flare_img = to_tensor(reflective_flare_img)
	scattering_flare_img = to_tensor(scattering_flare_img)
	light_source_img = to_tensor(light_source_img)
	flare_img = torch.clamp(scattering_flare_img+reflective_flare_img, min=0, max=1)
	light_source_img = torch.clamp(light_source_img, min=0, max=1)
	flare_img = remove_background(flare_img)
	both_images = torch.cat((light_source_img.unsqueeze(0), flare_img.unsqueeze(0)),0)
	both_images = transform_flare(both_images)
	flare_img =  both_images[1]
	light_source_img = both_images[0]
	flare_img = color_jitter(flare_img)
	light_source_img = color_jitter(light_source_img)
	blur_transform = transforms.GaussianBlur(21,sigma=(0.1,3.0))
	flare_img=blur_transform(flare_img)
	light_source_img = blur_transform(light_source_img)
	flare_img=flare_img+flare_DC_offset
	flare_img=torch.clamp(flare_img,min=0,max=1)
	
	img_with_light_source=torch.clamp(img_HR+light_source_img, min=-1, max=1)

    #merge image
	flare_corrupted_img=flare_img+img_HR
	flare_corrupted_img=torch.clamp(flare_corrupted_img,min=-1,max=1)
	return adjust_gamma_reverse(img_with_light_source), adjust_gamma_reverse(flare_corrupted_img)


if __name__ == "__main__":
	img_HR = Image.open("/home/intern/ztw/ztw/Methods/Difflare/data/Flickr24K/83.jpg").convert("RGB")
	scattering_flare_img = Image.open("/home/intern/ztw/ztw/Methods/Difflare/data/Flare7Kpp/Flare7K/Scattering_Flare/Compound_Flare/000001.png").convert("RGB")
	light_source_img = Image.open("/home/intern/ztw/ztw/Methods/Difflare/data/Flare7Kpp/Flare7K/Scattering_Flare/Light_Source/000001.png").convert("RGB")
	reflective_flare_img = Image.open("/home/intern/ztw/ztw/Methods/Difflare/data/Flare7Kpp/Flare7K/Reflective_Flare/000001.png").convert("RGB")
	img_with_light_source, flare_corrupted_img = generate_flare(img_HR,scattering_flare_img,reflective_flare_img,light_source_img)
	torchvision.utils.save_image(img_with_light_source, 'img_with_light_source.png')
	torchvision.utils.save_image(flare_corrupted_img,"reflective_flare_img.png")
	print("1")