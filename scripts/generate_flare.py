import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision
import os.path as osp

import numpy as np
from PIL import Image
import glob
import random
import cv2


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
		print("Light source not found.")
		return (255,255)
	else:
		largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
		largestCC=largestCC.astype(int)
		properties = regionprops(largestCC, largestCC)
		weighted_center_of_mass = properties[0].weighted_centroid
		print("Light source detected in position: x:",int(weighted_center_of_mass[1]),",y:",int(weighted_center_of_mass[0]))
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

class Flare_Image_Loader(data.Dataset):
	def __init__(self, image_path ,transform_base=None,transform_flare=None,mask_type=None):
		self.ext = ['png','jpeg','jpg','bmp','tif']
		self.data_list=[]
		[self.data_list.extend(glob.glob(image_path + '/*.' + e)) for e in self.ext]
		self.flare_dict={}
		self.flare_list=[]
		self.flare_name_list=[]

		self.reflective_flag=False
		self.reflective_dict={}
		self.reflective_list=[]
		self.reflective_name_list=[]

		self.mask_type=mask_type #It is a str which may be None,"luminance" or "color"

		self.transform_base=transform_base
		self.transform_flare=transform_flare

		print("Base Image Loaded with examples:", len(self.data_list))

	def __getitem__(self, index):
		# load base image
		img_path=self.data_list[index]
		base_img= Image.open(img_path)
		gamma=np.random.uniform(1.8,2.2)
		to_tensor=transforms.ToTensor()
		adjust_gamma=RandomGammaCorrection(gamma)
		adjust_gamma_reverse=RandomGammaCorrection(1/gamma)
		color_jitter=transforms.ColorJitter(brightness=(0.8,3),hue=0.0)
		if self.transform_base is not None:
			base_img=to_tensor(base_img)
			base_img=adjust_gamma(base_img)
			base_img=self.transform_base(base_img)
		else:
			base_img=to_tensor(base_img)
			base_img=adjust_gamma(base_img)
			base_img=base_img.permute(2,0,1)
		sigma_chi=0.01*np.random.chisquare(df=1)
		base_img=Normal(base_img,sigma_chi).sample()
		gain=np.random.uniform(1,1.2)
		flare_DC_offset=np.random.uniform(-0.02,0.02)
		base_img=gain*base_img
		base_img=torch.clamp(base_img,min=0,max=1)

		light_pos=plot_light_pos(base_img,0.97**gamma)
		
		light_pos=[light_pos[0]-256,light_pos[1]-256]
		#traslate=TranslationTransform(light_pos)
		
		transform_flare=transforms.Compose([transforms.RandomHorizontalFlip(),
							  transforms.RandomVerticalFlip(),
                              transforms.RandomAffine(degrees=(0,360),scale=(0.8,1.5),translate=(0,0),shear=(-20,20)),
							  TranslationTransform(light_pos),
                              transforms.CenterCrop((512,512)),
                              ])
		


		#load flare image
		flare_path=random.choice(self.flare_list)
		flare_img =Image.open(flare_path)
		idx = flare_path.split('/',-1)[-1]
		#load light source image
		single_light_source_path = osp.join(light_source_path,idx)
		light_source_img = Image.open(single_light_source_path)
		light_source_img = to_tensor(light_source_img)
		light_source_img = adjust_gamma(light_source_img)
		light_source_img = remove_background(light_source_img)
		# if self.transform_flare is not None:
		# 	light_source_img=self.transform_flare(light_source_img)
		# else:
		# 	light_source_img=transform_flare(light_source_img)
		

		if self.reflective_flag:
			reflective_path=random.choice(self.reflective_list)
			reflective_img = Image.open(reflective_path)


		flare_img=to_tensor(flare_img)
		flare_img=adjust_gamma(flare_img)
		
		if self.reflective_flag:
			reflective_img=to_tensor(reflective_img)
			reflective_img=adjust_gamma(reflective_img)
			flare_img = torch.clamp(flare_img+reflective_img,min=0,max=1)

		flare_img=remove_background(flare_img)

		both_images = torch.cat((light_source_img.unsqueeze(0), flare_img.unsqueeze(0)),0)

		if self.transform_flare is not None:
			both_images=self.transform_flare(both_images)
		else:
			both_images=transform_flare(both_images)
		
		light_source_img = both_images[0]
		flare_img = both_images[1]
		

		#change color
		light_source_img=color_jitter(light_source_img)

		#light source blur
		blur_transform=transforms.GaussianBlur(21,sigma=(0.1,3.0))
		light_source_img=blur_transform(light_source_img)
		light_source_img=light_source_img+flare_DC_offset
		light_source_img=torch.clamp(light_source_img,min=0,max=1)

		#merge gt	
		base_img=light_source_img+base_img
		base_img=torch.clamp(base_img,min=0,max=1)

		#change color
		flare_img=color_jitter(flare_img)

		#flare blur
		blur_transform=transforms.GaussianBlur(21,sigma=(0.1,3.0))
		flare_img=blur_transform(flare_img)
		flare_img=flare_img+flare_DC_offset
		flare_img=torch.clamp(flare_img,min=0,max=1)

		#merge image	
		merge_img=flare_img+base_img
		merge_img=torch.clamp(merge_img,min=0,max=1)

		if self.mask_type==None:
			return adjust_gamma_reverse(base_img),adjust_gamma_reverse(flare_img),adjust_gamma_reverse(merge_img),gamma
		elif self.mask_type=="luminance":
			#calculate mask (the mask is 3 channel)
			one = torch.ones_like(base_img)
			zero = torch.zeros_like(base_img)

			luminance=0.3*flare_img[0]+0.59*flare_img[1]+0.11*flare_img[2]
			threshold_value=0.99**gamma
			flare_mask=torch.where(luminance >threshold_value, one, zero)

		elif self.mask_type=="color":
			one = torch.ones_like(base_img)
			zero = torch.zeros_like(base_img)

			threshold_value=0.99**gamma
			flare_mask=torch.where(merge_img >threshold_value, one, zero)
		return adjust_gamma_reverse(base_img),adjust_gamma_reverse(flare_img),adjust_gamma_reverse(merge_img),flare_mask,gamma

	def __len__(self):
		return len(self.data_list)
	
	def load_scattering_flare(self,flare_name,flare_path):
		flare_list=[]
		[flare_list.extend(glob.glob(flare_path + '/*.' + e)) for e in self.ext]
		self.flare_name_list.append(flare_name)
		self.flare_dict[flare_name]=flare_list
		self.flare_list.extend(flare_list)
		len_flare_list=len(self.flare_dict[flare_name])
		if len_flare_list == 0:
			print("ERROR: scattering flare images are not loaded properly")
		else:
			print("Scattering Flare Image:",flare_name, " is loaded successfully with examples", str(len_flare_list))
		print("Now we have",len(self.flare_list),'scattering flare images')

	def load_reflective_flare(self,reflective_name,reflective_path):
		self.reflective_flag=True
		reflective_list=[]
		[reflective_list.extend(glob.glob(reflective_path + '/*.' + e)) for e in self.ext]
		self.reflective_name_list.append(reflective_name)
		self.reflective_dict[reflective_name]=reflective_list
		self.reflective_list.extend(reflective_list)
		len_reflective_list=len(self.reflective_dict[reflective_name])
		if len_reflective_list == 0:
			print("ERROR: reflective flare images are not loaded properly")
		else:
			print("Reflective Flare Image:",reflective_name, " is loaded successfully with examples", str(len_reflective_list))
		print("Now we have",len(self.reflective_list),'refelctive flare images')
	
if __name__ == "__main__":
	
	# import torchvision.transforms as transforms
    # from data_loader import Flare_Image_Loader

    transform_base=transforms.Compose([transforms.RandomCrop((512,512),pad_if_needed=True,padding_mode='reflect'),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip()
                                ])

    transform_flare=transforms.Compose([transforms.RandomAffine(degrees=(0,360),scale=(0.8,1.5),translate=(300/1440,300/1440),shear=(-20,20)),
                                transforms.CenterCrop((512,512)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip()
                                ])
    light_source_path = '/root/autodl-tmp/Difflare/data/Flare7Kpp/Flare7K/Scattering_Flare/Light_Source'
    flare_image_loader=Flare_Image_Loader('/root/autodl-tmp/Difflare/data/Flickr24K',transform_base,transform_flare)
    flare_image_loader.load_scattering_flare('Flare7K','/root/autodl-tmp/Difflare/data/Flare7Kpp/Flare7K/Scattering_Flare/Compound_Flare')
    flare_image_loader.load_reflective_flare('Flare7K','/root/autodl-tmp/Difflare/data/Flare7Kpp/Flare7K/Reflective_Flare')
    for images in range(len(flare_image_loader)):
        gt, flare, merge, mask = flare_image_loader[images]
        torchvision.utils.save_image(gt, "/root/autodl-tmp/Difflare/data/output_flare/gts/"+str(images)+".jpg")
        torchvision.utils.save_image(flare, "/root/autodl-tmp/Difflare/data/output_flare/flares/"+str(images)+".jpg")
        torchvision.utils.save_image(merge, "/root/autodl-tmp/Difflare/data/output_flare/merges/"+str(images)+".jpg")
 #       torchvision.utils.save_image(mask, "/root/autodl-tmp/StableSR/data/output_flare/masks/"+str(images)+".jpg")