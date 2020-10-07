import pydensecrf.densecrf as dcrf
import numpy as np
class CRF(object):
	def __init__(self,
				pos_xy_std=3,
				pos_w=3,
				bi_xy_std=80,
				bi_rgb_std=13,
				bi_w=10,
				maxiter=10,
				 scale_factor=1.0):
		self.pos_xy_std = pos_xy_std
		self.pos_w = pos_w
		self.bi_xy_std = bi_xy_std
		self.bi_rgb_std = bi_rgb_std
		self.bi_w = bi_w
		self.maxiter = maxiter
		self.scale_factor = scale_factor

	def inference(self, im, unary):
		H, W = im.shape[:2]
		C = unary.shape[0]
		d = dcrf.DenseCRF2D(W, H, C)
		d.setUnaryEnergy(-unary.reshape(C, -1))
		d.addPairwiseGaussian(sxy=(self.pos_xy_std/self.scale_factor, self.pos_xy_std/self.scale_factor),
							  compat=self.pos_w, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

		d.addPairwiseBilateral(sxy=(self.bi_xy_std/self.scale_factor, self.bi_xy_std/self.scale_factor),
							   srgb=(self.bi_rgb_std, self.bi_rgb_std, self.bi_rgb_std), rgbim=im.astype(np.uint8),
                           compat=self.bi_w,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
		prediction = np.array(d.inference(self.maxiter), dtype=np.float32).reshape((C, H, W))

		return prediction