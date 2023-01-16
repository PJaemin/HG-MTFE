import os
import sys
import torchvision
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageOps
import glob
import random
import Metrics
import cv2
import time

random.seed(123)


def get2Dhist(img):
	U = np.zeros((255, 255))
	tmp_k = np.array(range(1, 256))
	for layer in range(1, 256):
		U[:, layer - 1] = np.minimum(tmp_k, 256 - layer) - np.maximum(tmp_k - layer, 0)
	# alpha = 2.5
	R, C = img.shape
	# print(R)
	# print(C)
	if R == 255 and C == 255:
		h2D_in = img
	else:
		in_Y = img

		# % unordered 2D histogram acquisition
		h2D_in = np.zeros((256, 256))

		for j in range(1, R + 1):
			for i in range(1, R + 1):
				ref = in_Y[j - 1, i - 1]

				if j != R:
					trg = in_Y[j, i - 1]
				if i != C:
					trg = in_Y[j - 1, i]

				h2D_in[np.maximum(trg, ref), np.minimum(trg, ref)] = h2D_in[np.maximum(trg, ref), np.minimum(trg, ref)] + 1
	return h2D_in




class input_loader(data.Dataset):

	def __init__(self, image_path):
		self.image_path = image_path
		self.in_files = self.list_files(os.path.join(image_path, 'input'))

		if not os.path.exists('./data/train_data/curves'):
			os.makedirs('./data/train_data/curves')
			print('Created folder: ./data/train_data/curves')
			self.get_transFunction(image_path)

		if not os.path.exists('./data/train_data/2Dhist'):
			os.makedirs('./data/train_data/2Dhist')
			print('Created folder: ./data/train_data/2Dhist')
			self.get_2Dhist()

	def get_2Dhist(self):
		count = 0
		for file_name in self.in_files:
			print('\r[%d/%d] Creating 2D histograms...' % (count,len(self.in_files)), end='')
			src = cv2.imread(file_name)
			src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
			hist_s = np.zeros((3, 256, 256))

			for (j, color) in enumerate(("red", "green", "blue")):
				S = src[..., j]
				hist_s[j, ...] = get2Dhist(S)
				hist_s[j, ...] = hist_s[j, ...] / np.sum(hist_s[j,...])

			np.save('./data/train_data/2Dhist/' + os.path.splitext(os.path.basename(file_name))[0], hist_s)
			count += 1

	def get_transFunction(self, image_path):
		count = 0
		for file_name in self.in_files:
			print('\r[%d/%d] Creating transformation functions...' % (count,len(self.in_files)), end='')
			src = cv2.imread(file_name)
			gt = cv2.imread(image_path + 'gt/' + os.path.split(file_name)[-1])

			src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
			gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

			rCur = np.zeros(256)
			gCur = np.zeros(256)
			bCur = np.zeros(256)

			for (j, color) in enumerate(("red", "green", "blue")):

				S = src[..., j]
				G = gt[..., j]

				hist_s, _ = np.histogram(S.flatten(), 256, [0, 256])
				hist_g, _ = np.histogram(G.flatten(), 256, [0, 256])

				hist_s = hist_s / np.sum(hist_s)
				hist_g = hist_g / np.sum(hist_g)

				cdf_s = np.zeros(256)
				cdf_g = np.zeros(256)

				cdf_s[0] = hist_s[0]
				cdf_g[0] = hist_g[0]

				for t in range(1, 256):
					cdf_s[t] = cdf_s[t - 1] + hist_s[t]
					cdf_g[t] = cdf_g[t - 1] + hist_g[t]

				if color == "red":
					for m in range(256):
						tmp = 1
						for i in range(256):
							tmp1 = abs(cdf_s[m] - cdf_g[i])
							if tmp > tmp1:
								tmp = tmp1
								rCur[m] = i
				elif color == "green":
					for m in range(256):
						tmp = 1
						for i in range(256):
							tmp1 = abs(cdf_s[m] - cdf_g[i])
							if tmp > tmp1:
								tmp = tmp1
								gCur[m] = i
				else:
					for m in range(256):
						tmp = 1
						for i in range(256):
							tmp1 = abs(cdf_s[m] - cdf_g[i])
							if tmp > tmp1:
								tmp = tmp1
								bCur[m] = i

			rCur = np.expand_dims(rCur,0)
			gCur = np.expand_dims(gCur,0)
			bCur = np.expand_dims(bCur,0)

			curve = np.stack([rCur, gCur, bCur],axis=0)
			curve = curve.squeeze(1)
			curve = curve/255

			np.save('./data/train_data/curves/' + os.path.splitext(os.path.basename(file_name))[0], curve)
			count += 1

	def data_augment(self, inp, gt):
		a = random.randint(1, 4)
		if a == 1:
			return inp, gt
		# elif a == 2:
		# 	return inp.rotate(90, expand=True), gt.rotate(90, expand=True)
		elif a == 2:
			return inp.rotate(180, expand=True), gt.rotate(180, expand=True)
		# elif a == 4:
		# 	return inp.rotate(270, expand=True), gt.rotate(270, expand=True)
		# elif a == 5:
		# 	return ImageOps.flip(inp.rotate(90, expand=True)), ImageOps.flip(gt.rotate(90, expand=True))
		elif a == 3:
			return ImageOps.flip(inp.rotate(180, expand=True)), ImageOps.flip(gt.rotate(180, expand=True))
		# elif a == 7:
		# 	return ImageOps.flip(inp.rotate(270, expand=True)), ImageOps.flip(gt.rotate(270, expand=True))
		else:
			return ImageOps.flip(inp), ImageOps.flip(gt)

	def __getitem__(self, index):
		fname = os.path.split(self.in_files[index])[-1]
		data_low = Image.open(self.in_files[index])
		data_gt = Image.open(os.path.join(self.image_path, 'gt', fname))

		low = np.asarray(data_low)
		data_hist = np.zeros((3, 256))
		for i in range(3):
			S = low[..., i]
			data_hist[i, ...], _ = np.histogram(S.flatten(), 256, [0, 256])
			data_hist[i, ...] = data_hist[i, ...] / np.sum(data_hist[i, ...])

		data_input, data_gt = self.data_augment(data_low, data_gt)

		data_input = (np.asarray(data_input)/255.0)
		data_gt = (np.asarray(data_gt)/255.0)

		data_input = torch.from_numpy(data_input).float()
		data_gt = torch.from_numpy(data_gt).float()
		data_hist = torch.from_numpy(data_hist).float()

		# data_input = 2.0 * data_input - 1.0
		# data_gt = 2.0 * data_gt - 1.0
		# data_hist = 2.0 * data_hist - 1.0

		return data_input.permute(2,0,1), data_gt.permute(2,0,1), data_hist

	def __len__(self):
		return len(self.in_files)

	def list_files(self, in_path):
		files = []
		for (dirpath, dirnames, filenames) in os.walk(in_path):
			files.extend(filenames)
			break
		files = sorted([os.path.join(in_path, x) for x in files])
		return files