import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO

import torch
import torchvision.transforms as T


def main():
	parser = create_argument_parser()
	args = parser.parse_args()
	generate_coco_dataset(args)


def create_argument_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_root', type=str, default='datasets/COCO')
	parser.add_argument('--save_root', type=str, default='datasets/shp2gir_coco')
	parser.add_argument('--image_size', type=int, default=256, help='image size')
	parser.add_argument('--cat1', type=str, default='sheep', help='category 1')
	parser.add_argument('--cat2', type=str, default='giraffe', help='category 2')
	return parser


def generate_coco_dataset(args):
	"""Generate COCO dataset (train/val, A/B)"""
	args.data_root = Path(args.data_root)
	args.save_root = Path(args.save_root)
	args.save_root.mkdir()

	generate_coco_dataset_sub(args, 'train', 'A', args.cat1)
	generate_coco_dataset_sub(args, 'train', 'B', args.cat2)
	generate_coco_dataset_sub(args, 'val', 'A', args.cat1)
	generate_coco_dataset_sub(args, 'val', 'B', args.cat2)


def generate_coco_dataset_sub(args, idx1, idx2, cat):
	"""
	Subroutine for generating COCO dataset
		- idx1: train/val
		- idx2: A/B
		- cat: category
	"""
	data_path = args.data_root / '{}2017'.format(idx1)
	anno_path = args.data_root / 'annotations/instances_{}2017.json'.format(idx1)
	coco = COCO(anno_path)  # COCO API

	img_path = args.save_root / '{}{}'.format(idx1, idx2)
	seg_path = args.save_root / '{}{}_seg'.format(idx1, idx2)
	img_path.mkdir()
	seg_path.mkdir()

	cat_id = coco.getCatIds(catNms=cat)
	img_id = coco.getImgIds(catIds=cat_id)
	imgs = coco.loadImgs(img_id)

	pb = tqdm(total=len(imgs))
	pb.set_description('{}{}'.format(idx1, idx2))
	for img in imgs:
		ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_id)
		anns = coco.loadAnns(ann_ids)

		count = 0
		for i in range(len(anns)):
			seg = coco.annToMask(anns[i])
			seg = Image.fromarray(seg * 255)
			seg = resize(seg, args.image_size)
			if np.sum(np.asarray(seg)) > 0:
				seg.save(seg_path / '{}_{}.png'.format(pb.n, count))
				count += 1

		if count > 0:  # at least one instance exists
			img = Image.open(data_path / img['file_name'])
			img = resize(img, args.image_size)
			img.save(img_path / '{}.png'.format(pb.n))

		pb.update(1)
	pb.close()


def resize(img, size):
	return T.Compose([
		T.Resize(size),
		T.CenterCrop(size),
	])(img)


if __name__ == '__main__':
	main()
