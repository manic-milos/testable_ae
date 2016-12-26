import tensorflow as tf;
import numpy as np;
import math;
import upscaling;

class Autoencoder:
	def __init__(self,input_dim,represent_dim,prefix='',encW_init=None,
			  encb_init=None,decW_init=None,decb_init=None):
		self.input_dim=input_dim;
		self.represent_dim=represent_dim;
		self.x = tf.placeholder(tf.float32, [None, input_dim]);
		
		#decW initialization
		if(decW_init is None):
			decW_init=tf.random_normal(
				[represent_dim,input_dim],
				-math.sqrt(6.0/(input_dim+represent_dim)),
				math.sqrt(6.0/(input_dim+represent_dim)));
		
		self.decW=tf.Variable(
			initial_value=decW_init,
				dtype=tf.float32,
				name=prefix+'decW');
		
		#encW initialization
		if(encW_init is None):
			encW_init=tf.random_normal(
				[input_dim, represent_dim],
				-math.sqrt(6.0/(input_dim+represent_dim)),
				math.sqrt(6.0/(input_dim+represent_dim)));
		
		self.encW=tf.Variable(
			initial_value=encW_init,dtype=tf.float32,
			name=prefix+'encW');
		
		#encb initialization
		if(encb_init is None):
			encb_init=tf.zeros([represent_dim]);
			
		self.encb=tf.Variable(encb_init,
						dtype=tf.float32,
			name=prefix+'encb');
		
		#decb initialization
		if(decb_init is None):
			decb_init=tf.zeros([input_dim])
			
		self.decb=tf.Variable(
			decb_init,
			dtype=tf.float32,
			name=prefix+'decb');
		
		
		self.representation=tf.nn.sigmoid(
			tf.matmul(self.x,self.encW) + self.encb);
		self.hidden_weights=self.encW;
		self.reconstruction=tf.nn.sigmoid(
			tf.matmul(self.representation,self.decW)+self.decb);
		self.cost=tf.nn.l2_loss(self.x-self.reconstruction);
		self.loss_per_pixel=tf.reduce_mean(
			tf.abs(self.x-self.reconstruction));
		pixel_average=tf.reduce_mean(tf.abs(self.x));
		self.reference_model_error=tf.reduce_mean(
			tf.abs(self.x-pixel_average));
		self.maxerr=tf.reduce_max(tf.abs(self.x-self.reconstruction));
		self.minerr=tf.reduce_min(tf.abs(self.x-self.reconstruction));
		
	def getOptimizationVarsList(self):
		return [self.encW,self.decW,self.encb,self.decb];
	
	
	def getCost(self,sess,x):
		
		return sess.run(self.cost,
			feed_dict={
			self.x:x
			});
	
	
	def getCostPerPixel(self,sess,x):
		
		return sess.run(self.loss_per_pixel,
			feed_dict={
			self.x:x
			});
			
	def upscale(self,lowres_coords,highres_coords,sess,prefix=''):
		
		input_dim=self.input_dim*2;
		represent_dim=self.represent_dim;
		encW=sess.run(self.encW);
		decW=sess.run(self.decW);
		encb=sess.run(self.encb);
		decb=sess.run(self.decb);
		
		
		print "upscaling encoder...";
		high_W=np.zeros([input_dim,represent_dim]);
		i_num=np.zeros(len(lowres_coords));
		for i in range(input_dim):
			notNa,li=upscaling.get_lowres_coords(highres_coords[i][0],
				highres_coords[i][1],lowres_coords,highres_coords);
			i_num[li]+=1;
			for j in range(represent_dim):
				high_W[i][j]=encW[li][j];
		print "upscaling encoder finished...";
		print "adjusting encoder weights...";
		for i in range (input_dim):
			o,li=upscaling.get_lowres_coords(highres_coords[i][0],
				highres_coords[i][1],lowres_coords,highres_coords);
			for j in range(represent_dim):
					high_W[i][j]/=i_num[li];
		print "adjusting encoder weights...";
		high_encW=high_W;
		
		#decoder weights
		high_W=np.zeros([represent_dim,input_dim]);
		for i in range(input_dim):
			o,li=upscaling.get_lowres_coords(highres_coords[i][0],
				highres_coords[i][1],lowres_coords,highres_coords);
			for j in range(represent_dim):
				high_W[j][i]=decW[j][li];
		high_decW=high_W;
		
		#decoder bias
		dec_b_init=np.zeros([input_dim]);
		for i in range(input_dim):
			o,li=upscaling.get_lowres_coords(highres_coords[i][0],
				highres_coords[i][1],lowres_coords,highres_coords);
			dec_b_init[i]=decb[li];
		
		high_ae=Autoencoder(input_dim,represent_dim,
					  prefix='',encW_init=high_encW,
			  encb_init=encb,decW_init=high_decW,decb_init=dec_b_init);
		
		return high_ae;
		
		
		
		
		
		
		
		
		
		
		
		
		