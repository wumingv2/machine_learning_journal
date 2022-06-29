
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.engine.topology import Layer
#some loss functions

class MaskedMultiCrossEntropy(object):

	def loss(self, y_true, y_pred):
		vec = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true, dim=2)
		mask = tf.equal(y_true[:,:,0], -1)

		zer = tf.zeros_like(vec)
		loss = tf.where(mask, x=zer, y=vec)
		return loss

class MaskedMultiCrossEntropy_sce(object):

	def loss(self, y_true, y_pred):
		y_true_1 = y_true
		y_pred_1 = y_pred

		y_true_2 = y_true
		y_pred_2 = y_pred
		vec = -tf.reduce_sum(y_true_1 * tf.log(tf.clip_by_value(y_pred_1, 1e-7, 1.0)), axis=-1)#[?,3]
		vec2 = -tf.reduce_sum(y_pred_2 * tf.log(tf.clip_by_value(y_true_2, 1e-4, 1.0)), axis=-1)#[?,3]

		mask = tf.equal(y_true_1[:,:,0], -1)
		zer = tf.zeros_like(vec)
		loss1 = tf.where(mask, x=zer, y=vec)
		loss2 = tf.where(mask, x=zer, y=vec2)
		loss= 0.1* loss1+ 1*loss2 #tune the hyperparameter
		return loss

class MaskedMultiCrossEntropy_gce(object):

	def loss(self, y_true, y_pred):
		q = 0.7
		t_loss = (1 - tf.pow(tf.reduce_sum(tf.abs(y_true * y_pred), axis=-1), q)) / q ##tune the hyperparameter
		mask = tf.equal(y_true[:, :, 0], -1)
		zer = tf.zeros_like(t_loss)
		loss = tf.where(mask, x=zer, y=t_loss)
		return loss

class MaskedMultiCrossEntropy_cl(object):

	def loss(self, y_true, y_pred):
		vec = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true, dim=1)
		mask = tf.equal(y_true[:,0,:], -1)

		zer = tf.zeros_like(vec)
		loss = tf.where(mask, x=zer, y=vec)
		return loss

class MaskedMultiMSE(object):
		
	def loss(self, y_true, y_pred):
		vec = K.square(y_pred - y_true)
		mask = tf.equal(y_true[:,:], 999999999)
		zer = tf.zeros_like(vec)
		loss = tf.where(mask, x=zer, y=vec)
		return loss

	def __init__(self, num_classes):
		self.num_classes = num_classes

	def loss(self, y_true, y_pred):
		mask_missings = tf.equal(y_true, -1)
		mask_padding = tf.equal(y_true, 0)

		# convert targets to one-hot enconding and transpose
		y_true = tf.transpose(tf.one_hot(tf.cast(y_true, tf.int32), self.num_classes, axis=-1), [0,1,3,2])

		# masked cross-entropy
		vec = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true, dim=2)
		zer = tf.zeros_like(vec)
		vec = tf.where(mask_missings, x=zer, y=vec)
		vec = tf.where(mask_padding, x=zer, y=vec)
		loss = tf.reduce_mean(vec, axis=-1)
		return loss

	def __init__(self, num_classes, num_annotators, pi_prior=0.01):
		self.num_classes = num_classes
		self.num_annotators = num_annotators
		self.pi_prior = pi_prior
		
		# initialize pi_est (annotators' estimated confusion matrices) wit identities
		self.pi_est = np.zeros((self.num_classes,self.num_classes,self.num_annotators), dtype=np.float32)
		for r in xrange(self.num_annotators):
			self.pi_est[:,:,r] = np.eye(self.num_classes) + self.pi_prior
			self.pi_est[:,:,r] /= np.sum(self.pi_est[:,:,r], axis=1)
			
		self.init_suff_stats()
			
	def init_suff_stats(self):
		# initialize suff stats for M-step
		self.suff_stats = self.pi_prior * tf.ones((self.num_annotators,self.num_classes,self.num_classes))
		
	def loss_fc(self, y_true, y_pred):
		y_true = tf.cast(y_true, tf.int32)

		#y_pred += 0.01
		#y_pred /= tf.reduce_sum(y_pred, reduction_indices=len(y_pred.get_shape()) - 1, keep_dims=True)

		#y_pred = tf.where(tf.less(y_pred, 0.001), 
		#                        #0.01 * tf.ones_like(y_pred), 
		#                        0.001 + y_pred, 
		#                        y_pred)
		#y_pred += 0.01 # y_pred cannot be zero!
		eps = 1e-3
		#y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
		y_pred = tf.clip_by_value(y_pred, eps, 9999999999)


		# E-step
		adjustment_factor = tf.ones_like(y_pred)
		for r in xrange(self.num_annotators):
			adj = tf.where(tf.equal(y_true[:,r], -1), 
								tf.ones_like(y_pred), 
								tf.gather(tf.transpose(self.pi_est[:,:,r]), y_true[:,r]))
			adjustment_factor = tf.multiply(adjustment_factor, adj)
			
		res = tf.multiply(adjustment_factor, y_pred)
		y_agg = res / tf.expand_dims(tf.reduce_sum(res, axis=1), 1)

		loss = -tf.reduce_sum(y_agg * tf.log(y_pred), reduction_indices=[1])
		
		# update suff stats
		upd_suff_stats = []
		for r in xrange(self.num_annotators):
			#print r
			suff_stats = []
			normalizer = tf.zeros_like(y_pred)
			for c in xrange(self.num_classes):
				suff_stats.append(tf.reduce_sum(tf.where(tf.equal(y_true[:,r], c), 
									y_agg,
									tf.zeros_like(y_pred)), axis=0))
			upd_suff_stats.append(suff_stats)
		upd_suff_stats = tf.stack(upd_suff_stats)
		self.suff_stats += upd_suff_stats

		return loss
	
	def m_step(self):
		#print "M-step"
		self.pi_est = tf.transpose(self.suff_stats / tf.expand_dims(tf.reduce_sum(self.suff_stats, axis=2), 2), [1, 2, 0])
		
		return self.pi_est




	def __init__(self, loss):
		self.loss = loss
		
	def on_epoch_begin(self, epoch, logs=None):
		self.loss.init_suff_stats()
		
	def on_epoch_end(self, epoch, logs=None):
		# run M-step
		self.model.pi = self.loss.m_step()


