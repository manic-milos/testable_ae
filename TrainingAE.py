from Autoencoder import *;
import tensorflow as tf;
import numpy as np;
import math;
import time;

def train(instances,
		  ae,
		  optimizer,
		  start_epoch,n_epochs,
		  batch_size,
		  sess):
	
	

	#setting random seeds
	np.random.seed(3);
	
	print "initializing sets...";
	trainingset=instances[1:int(0.8*len(instances))];
	trainingset=np.random.permutation(np.array(trainingset));
	validationset=instances[int(0.8*len(instances)):int(0.9*len(instances))];
	testset_beggining=int(0.9*len(instances))
	testset=instances[testset_beggining:];
	print "sets initialized...";
	
	
	

	
	
	print "training error(L2):",
	print ae.getCost(sess,trainingset),
	print "per pixel:",
	print ae.getCostPerPixel(sess,trainingset);
	print "validation error(L2):",
	print ae.getCost(sess,validationset),
	print "per pixel:",
	print ae.getCostPerPixel(sess,validationset);
	print "test error(L2):",
	print ae.getCost(sess,testset),
	print "per pixel:",
	print ae.getCostPerPixel(sess,testset);
	
	
	c=0;
	error=10000000000;
	
	#training
	start_time = time.time();
	
	for epoch_i in range(
		start_epoch, start_epoch+n_epochs):
		i_batch=0;
		max=0;
		print ("epoch "+str(epoch_i)+":\t"),
		for batch_i in range(len(trainingset) // batch_size):
			batch_xs= trainingset[i_batch:i_batch+batch_size];
			i_batch=i_batch+batch_size;
			o,c=sess.run(fetches=[optimizer,ae.cost],
				feed_dict={ae.x:batch_xs});
			if(c>max):
				max=c;
		c,cpp=sess.run(fetches=[ae.cost,ae.loss_per_pixel],
			feed_dict={
			ae.x:trainingset
			});
		cppv=sess.run(fetches=ae.loss_per_pixel,feed_dict={
			ae.x:validationset
			});
		print "%2.6f\t"%(c),
		print "%2.10f\t"%(cpp),
		print "%2.10f"%(cppv);
		if(epoch_i%100==0 and epoch_i!=start_epoch):
			print "test set error per pixel:",
			teppx=sess.run(ae.loss_per_pixel,
				feed_dict={
				ae.x:testset
				});
			print teppx;
	
	#calculate time elapsed
	print "time elapsed:",time.time()-start_time;
	return [trainingset,validationset,testset];