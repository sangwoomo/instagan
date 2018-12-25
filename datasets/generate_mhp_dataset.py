import argparse
import cv2
import math
import numpy as np
import scipy.io as sio
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import torch
import torchvision.transforms as T


def main():
	parser = create_argument_parser()
	args = parser.parse_args()
	generate_ccp_dataset(args)

def create_argument_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_root', type=str, default='datasets/LV-MHP-v1')
	parser.add_argument('--save_root', type=str, default='datasets/pants2skirt_mhp')
	parser.add_argument('--cat1', type=str, default='pants', help='category 1')
	parser.add_argument('--cat2', type=str, default='skirt', help='category 2')
	parser.add_argument('--size_h', type=float, default=450, help='height')
	parser.add_argument('--size_w', type=float, default=300, help='width')
	parser.add_argument('--no_skip_horizontal', action='store_true', help='do *not* skip horizontal images')
	return parser

def generate_ccp_dataset(args):
	"""
	Generate COCO dataset (train/val, A/B)
	"""
	args.data_root = Path(args.data_root)
	args.img_root = args.data_root / 'images'
	args.ann_root = args.data_root / 'annotations'

	args.save_root = Path(args.save_root)
	args.save_root.mkdir()

	generate_mhp_dataset(args, 'train', 'A', get_cat_id(args.cat1))
	generate_mhp_dataset(args, 'train', 'B', get_cat_id(args.cat2))
	generate_mhp_dataset(args, 'test', 'A', get_cat_id(args.cat1))
	generate_mhp_dataset(args, 'test', 'B', get_cat_id(args.cat2))

def generate_mhp_dataset(args, phase, domain, cat):
	img_path = args.save_root / '{}{}'.format(phase, domain)
	seg_path = args.save_root / '{}{}_seg'.format(phase, domain)
	img_path.mkdir()
	seg_path.mkdir()

	idx_path = args.data_root / '{}_list.txt'.format(phase)
	f = idx_path.open()
	idxs = f.readlines()

	pb = tqdm(total=len(idxs))
	pb.set_description('{}{}'.format(phase, domain))
	for idx in idxs:
		count = 0  # number of instances
		id = idx.split('.')[0]  # before extension
		for ann_path in args.ann_root.iterdir():
			if ann_path.name.split('_')[0] == id:
				ann = cv2.imread(str(ann_path))
				if not args.no_skip_horizontal:
					if ann.shape[1] > ann.shape[0]:
						continue  # skip horizontal image
				if np.isin(ann, cat).sum() > 0:
					seg = (ann == cat).astype('uint8')  # get segment of given category
					seg = Image.fromarray(seg * 255)
					seg = resize_and_crop(seg, [args.size_w, args.size_h])  # resize and crop
					if np.sum(np.asarray(seg)) > 0:
						seg.save(seg_path / '{}_{}.png'.format(id, count))
						count += 1
		if count > 0:
			# img = Image.open(args.img_root / '{}.jpg'.format(id))
			# PIL fails to open Image -> hence, open with cv2
			# https://stackoverflow.com/questions/48944819/image-open-gives-error-cannot-identify-image-file
			img = cv2.imread(str(args.img_root / '{}.jpg'.format(id)))
			# convert cv2 image to PIL image format
			# https://stackoverflow.com/questions/43232813/convert-opencv-image-format-to-pil-image-format?noredirect=1&lq=1
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img = Image.fromarray(img)
			img = resize_and_crop(img, [args.size_w, args.size_h])
			img.save(img_path / '{}.png'.format(id))

		pb.update(1)

	pb.close()

def get_cat_id(cat):
	return {
		'background': 0,
		'hat': 1,
		'hair': 2,
		'sunglass': 3,
		'upper-clothes': 4,
		'skirt': 5,
		'pants': 6,
		'dress': 7,
		'belt': 8,
		'left-shoe': 9,
		'right-shoe': 10,
		'face': 11,
		'left-leg': 12,
		'right-leg': 13,
		'left-arm': 14,
		'right-arm': 15,
		'bag': 16,
		'scarf': 17,
		'torso-skin': 18,
	}[cat]

def resize_and_crop(img, size):
	src_w, src_h = img.size
	tgt_w, tgt_h = size
	ceil_w = math.ceil((src_w / src_h) * tgt_h)
	return T.Compose([
		T.Resize([tgt_h, ceil_w]),
		T.CenterCrop([tgt_h, tgt_w]),
	])(img)

if __name__ == '__main__':
	main()