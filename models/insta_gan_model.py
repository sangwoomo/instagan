import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import copy


class InstaGANModel(BaseModel):
	def name(self):
		return 'InstaGANModel'

	@staticmethod
	def modify_commandline_options(parser, is_train=True):
		# default CycleGAN did not use dropout
		parser.set_defaults(no_dropout=True)
		parser.add_argument('--set_order', type=str, default='decreasing', help='order of segmentation')
		parser.add_argument('--ins_max', type=int, default=4, help='maximum number of instances to forward')
		parser.add_argument('--ins_per', type=int, default=2, help='number of instances to forward, for one pass')
		if is_train:
			parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
			parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
			parser.add_argument('--lambda_idt', type=float, default=1.0, help='use identity mapping. Setting lambda_idt other than 0 has an effect of scaling the weight of the identity mapping loss')
			parser.add_argument('--lambda_ctx', type=float, default=1.0, help='use context preserving. Setting lambda_ctx other than 0 has an effect of scaling the weight of the context preserving loss')

		return parser

	def initialize(self, opt):
		BaseModel.initialize(self, opt)

		self.ins_iter = self.opt.ins_max // self.opt.ins_per  # number of forward iteration

		# specify the training losses you want to print out. The program will call base_model.get_current_losses
		self.loss_names = ['D_A', 'G_A', 'cyc_A', 'idt_A', 'ctx_A', 'D_B', 'G_B', 'cyc_B', 'idt_B', 'ctx_B']
		# specify the images you want to save/display. The program will call base_model.get_current_visuals
		visual_names_A_img = ['real_A_img', 'fake_B_img', 'rec_A_img']
		visual_names_B_img = ['real_B_img', 'fake_A_img', 'rec_B_img']
		visual_names_A_seg = ['real_A_seg', 'fake_B_seg', 'rec_A_seg']
		visual_names_B_seg = ['real_B_seg', 'fake_A_seg', 'rec_B_seg']
		self.visual_names = visual_names_A_img + visual_names_A_seg + visual_names_B_img + visual_names_B_seg
		# specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
		if self.isTrain:
			self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
		else:
			self.model_names = ['G_A', 'G_B']

		# load/define networks
		# The naming conversion is different from those used in the paper
		# Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
		self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
		self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
		if self.isTrain:
			use_sigmoid = opt.no_lsgan
			self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
			self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

		if self.isTrain:
			self.fake_A_pool = ImagePool(opt.pool_size)
			self.fake_B_pool = ImagePool(opt.pool_size)
			# define loss functions
			self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
			self.criterionCyc = torch.nn.L1Loss()
			self.criterionIdt = torch.nn.L1Loss()
			# initialize optimizers
			self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, itertools.chain(self.netG_A.parameters(), self.netG_B.parameters())), lr=opt.lr, betas=(opt.beta1, 0.999))
			self.optimizer_D = torch.optim.Adam(filter(lambda p: p.requires_grad, itertools.chain(self.netD_A.parameters(), self.netD_B.parameters())), lr=opt.lr, betas=(opt.beta1, 0.999))
			self.optimizers = []
			self.optimizers.append(self.optimizer_G)
			self.optimizers.append(self.optimizer_D)

	def select_masks(self, segs_batch):
		"""Select instance masks to use"""
		if self.opt.set_order == 'decreasing':
			return self.select_masks_decreasing(segs_batch)
		elif self.opt.set_order == 'random':
			return self.select_masks_random(segs_batch)
		else:
			raise NotImplementedError('Set order name [%s] is not recognized' % self.opt.set_order)

	def select_masks_decreasing(self, segs_batch):
		"""Select masks in decreasing order"""
		ret = list()
		for segs in segs_batch:
			mean = segs.mean(-1).mean(-1)
			m, i = mean.topk(self.opt.ins_max)
			ret.append(segs[i, :, :])
		return torch.stack(ret)

	def select_masks_random(self, segs_batch):
		"""Select masks in random order"""
		ret = list()
		for segs in segs_batch:
			mean = (segs + 1).mean(-1).mean(-1)
			m, i = mean.topk(self.opt.ins_max)
			num = min(len(mean.nonzero()), self.opt.ins_max)
			reorder = np.concatenate((np.random.permutation(num), np.arange(num, self.opt.ins_max)))
			ret.append(segs[i[reorder], :, :])
		return torch.stack(ret)

	def merge_masks(self, segs):
		"""Merge masks (B, N, W, H) -> (B, 1, W, H)"""
		ret = torch.sum((segs + 1)/2, dim=1, keepdim=True)  # (B, 1, W, H)
		return ret.clamp(max=1, min=0) * 2 - 1

	def get_weight_for_ctx(self, x, y):
		"""Get weight for context preserving loss"""
		z = self.merge_masks(torch.cat([x, y], dim=1))
		return (1 - z) / 2  # [-1,1] -> [1,0]

	def weighted_L1_loss(self, src, tgt, weight):
		"""L1 loss with given weight (used for context preserving loss)"""
		return torch.mean(weight * torch.abs(src - tgt))

	def split(self, x):
		"""Split data into image and mask (only assume 3-channel image)"""
		return x[:, :3, :, :], x[:, 3:, :, :]

	def set_input(self, input):
		AtoB = self.opt.direction == 'AtoB'
		self.real_A_img = input['A' if AtoB else 'B'].to(self.device)
		self.real_B_img = input['B' if AtoB else 'A'].to(self.device)
		real_A_segs = input['A_segs' if AtoB else 'B_segs']
		real_B_segs = input['B_segs' if AtoB else 'A_segs']
		self.real_A_segs = self.select_masks(real_A_segs).to(self.device)
		self.real_B_segs = self.select_masks(real_B_segs).to(self.device)
		self.real_A = torch.cat([self.real_A_img, self.real_A_segs], dim=1)
		self.real_B = torch.cat([self.real_B_img, self.real_B_segs], dim=1)
		self.real_A_seg = self.merge_masks(self.real_A_segs)  # merged mask
		self.real_B_seg = self.merge_masks(self.real_B_segs)  # merged mask
		self.image_paths = input['A_paths' if AtoB else 'B_paths']

	def forward(self, idx=0):
		N = self.opt.ins_per
		self.real_A_seg_sng = self.real_A_segs[:, N*idx:N*(idx+1), :, :]  # ith mask
		self.real_B_seg_sng = self.real_B_segs[:, N*idx:N*(idx+1), :, :]  # ith mask
		empty = -torch.ones(self.real_A_seg_sng.size()).to(self.device)  # empty image

		self.forward_A = (self.real_A_seg_sng + 1).sum() > 0  # check if there are remaining instances
		self.forward_B = (self.real_B_seg_sng + 1).sum() > 0  # check if there are remaining instances

		# forward A
		if self.forward_A:
			self.real_A_sng = torch.cat([self.real_A_img_sng, self.real_A_seg_sng], dim=1)
			self.fake_B_sng = self.netG_A(self.real_A_sng)
			self.rec_A_sng = self.netG_B(self.fake_B_sng)

			self.fake_B_img_sng, self.fake_B_seg_sng = self.split(self.fake_B_sng)
			self.rec_A_img_sng, self.rec_A_seg_sng = self.split(self.rec_A_sng)
			fake_B_seg_list = self.fake_B_seg_list + [self.fake_B_seg_sng]  # not detach
			for i in range(self.ins_iter - idx - 1):
				fake_B_seg_list.append(empty)

			self.fake_B_seg_mul = torch.cat(fake_B_seg_list, dim=1)
			self.fake_B_mul = torch.cat([self.fake_B_img_sng, self.fake_B_seg_mul], dim=1)

		# forward B
		if self.forward_B:
			self.real_B_sng = torch.cat([self.real_B_img_sng, self.real_B_seg_sng], dim=1)
			self.fake_A_sng = self.netG_B(self.real_B_sng)
			self.rec_B_sng = self.netG_A(self.fake_A_sng)

			self.fake_A_img_sng, self.fake_A_seg_sng = self.split(self.fake_A_sng)
			self.rec_B_img_sng, self.rec_B_seg_sng = self.split(self.rec_B_sng)
			fake_A_seg_list = self.fake_A_seg_list + [self.fake_A_seg_sng]  # not detach
			for i in range(self.ins_iter - idx - 1):
				fake_A_seg_list.append(empty)

			self.fake_A_seg_mul = torch.cat(fake_A_seg_list, dim=1)
			self.fake_A_mul = torch.cat([self.fake_A_img_sng, self.fake_A_seg_mul], dim=1)

	def test(self):
		self.real_A_img_sng = self.real_A_img
		self.real_B_img_sng = self.real_B_img
		self.fake_A_seg_list = list()
		self.fake_B_seg_list = list()
		self.rec_A_seg_list = list()
		self.rec_B_seg_list = list()

		# sequential mini-batch translation
		for i in range(self.ins_iter):
			# forward
			with torch.no_grad():  # no grad
				self.forward(i)

			# update setting for next iteration
			self.real_A_img_sng = self.fake_B_img_sng.detach()
			self.real_B_img_sng = self.fake_A_img_sng.detach()
			self.fake_A_seg_list.append(self.fake_A_seg_sng.detach())
			self.fake_B_seg_list.append(self.fake_B_seg_sng.detach())
			self.rec_A_seg_list.append(self.rec_A_seg_sng.detach())
			self.rec_B_seg_list.append(self.rec_B_seg_sng.detach())

			# save visuals
			if i == 0:  # first
				self.rec_A_img = self.rec_A_img_sng
				self.rec_B_img = self.rec_B_img_sng
			if i == self.ins_iter - 1:  # last
				self.fake_A_img = self.fake_A_img_sng
				self.fake_B_img = self.fake_B_img_sng
				self.fake_A_seg = self.merge_masks(self.fake_A_seg_mul)
				self.fake_B_seg = self.merge_masks(self.fake_B_seg_mul)
				self.rec_A_seg = self.merge_masks(torch.cat(self.rec_A_seg_list, dim=1))
				self.rec_B_seg = self.merge_masks(torch.cat(self.rec_B_seg_list, dim=1))

	def backward_G(self):
		lambda_A = self.opt.lambda_A
		lambda_B = self.opt.lambda_B
		lambda_idt = self.opt.lambda_idt
		lambda_ctx = self.opt.lambda_ctx

		# backward A
		if self.forward_A:
			self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B_mul), True)
			self.loss_cyc_A = self.criterionCyc(self.rec_A_sng, self.real_A_sng) * lambda_A
			self.loss_idt_B = self.criterionIdt(self.netG_B(self.real_A_sng), self.real_A_sng.detach()) * lambda_A * lambda_idt
			weight_A = self.get_weight_for_ctx(self.real_A_seg_sng, self.fake_B_seg_sng)
			self.loss_ctx_A = self.weighted_L1_loss(self.real_A_img_sng, self.fake_B_img_sng, weight=weight_A) * lambda_A * lambda_ctx
		else:
			self.loss_G_A = 0
			self.loss_cyc_A = 0
			self.loss_idt_B = 0
			self.loss_ctx_A = 0

		# backward B
		if self.forward_B:
			self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A_mul), True)
			self.loss_cyc_B = self.criterionCyc(self.rec_B_sng, self.real_B_sng) * lambda_B
			self.loss_idt_A = self.criterionIdt(self.netG_A(self.real_B_sng), self.real_B_sng.detach()) * lambda_B * lambda_idt
			weight_B = self.get_weight_for_ctx(self.real_B_seg_sng, self.fake_A_seg_sng)
			self.loss_ctx_B = self.weighted_L1_loss(self.real_B_img_sng, self.fake_A_img_sng, weight=weight_B) * lambda_B * lambda_ctx
		else:
			self.loss_G_B = 0
			self.loss_cyc_B = 0
			self.loss_idt_A = 0
			self.loss_ctx_B = 0

		# combined loss
		self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cyc_A + self.loss_cyc_B + self.loss_idt_A + self.loss_idt_B + self.loss_ctx_A + self.loss_ctx_B
		self.loss_G.backward()

	def backward_D_basic(self, netD, real, fake):
		# Real
		pred_real = netD(real)
		loss_D_real = self.criterionGAN(pred_real, True)
		# Fake
		pred_fake = netD(fake.detach())
		loss_D_fake = self.criterionGAN(pred_fake, False)
		# Combined loss
		loss_D = (loss_D_real + loss_D_fake) * 0.5
		# backward
		loss_D.backward()
		return loss_D

	def backward_D_A(self):
		fake_B = self.fake_B_pool.query(self.fake_B_mul)
		self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

	def backward_D_B(self):
		fake_A = self.fake_A_pool.query(self.fake_A_mul)
		self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

	def optimize_parameters(self):
		# init setting
		self.real_A_img_sng = self.real_A_img
		self.real_B_img_sng = self.real_B_img
		self.fake_A_seg_list = list()
		self.fake_B_seg_list = list()
		self.rec_A_seg_list = list()
		self.rec_B_seg_list = list()

		# sequential mini-batch translation
		for i in range(self.ins_iter):
			# forward
			self.forward(i)

			# G_A and G_B
			if self.forward_A or self.forward_B:
				self.set_requires_grad([self.netD_A, self.netD_B], False)
				self.optimizer_G.zero_grad()
				self.backward_G()
				self.optimizer_G.step()

			# D_A and D_B
			if self.forward_A or self.forward_B:
				self.set_requires_grad([self.netD_A, self.netD_B], True)
				self.optimizer_D.zero_grad()
				if self.forward_A:
					self.backward_D_A()
				if self.forward_B:
					self.backward_D_B()
				self.optimizer_D.step()

			# update setting for next iteration
			self.real_A_img_sng = self.fake_B_img_sng.detach()
			self.real_B_img_sng = self.fake_A_img_sng.detach()
			self.fake_A_seg_list.append(self.fake_A_seg_sng.detach())
			self.fake_B_seg_list.append(self.fake_B_seg_sng.detach())
			self.rec_A_seg_list.append(self.rec_A_seg_sng.detach())
			self.rec_B_seg_list.append(self.rec_B_seg_sng.detach())

			# save visuals
			if i == 0:  # first
				self.rec_A_img = self.rec_A_img_sng
				self.rec_B_img = self.rec_B_img_sng
			if i == self.ins_iter - 1:  # last
				self.fake_A_img = self.fake_A_img_sng
				self.fake_B_img = self.fake_B_img_sng
				self.fake_A_seg = self.merge_masks(self.fake_A_seg_mul)
				self.fake_B_seg = self.merge_masks(self.fake_B_seg_mul)
				self.rec_A_seg = self.merge_masks(torch.cat(self.rec_A_seg_list, dim=1))
				self.rec_B_seg = self.merge_masks(torch.cat(self.rec_B_seg_list, dim=1))
