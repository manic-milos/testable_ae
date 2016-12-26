
import data_loading as dl;
from Data import *
def load_data(map_schema_filename,maps_foldername):
	#data loading 

	#schema loading
	print "loading schema..."
	[coords,img_dims]=dl.load_schema(map_schema_filename);
	print "schema loaded";
	#end schema loading


	print "loading maps..."
	[instances,
		coords,
		img_dims,
		onlyfiles]=dl.load_nodes_with_schema(maps_foldername,
		map_schema_filename);
	print "loading maps completed";
	return Data(instances,coords,img_dims);

def load_data_config(config):
	config.data=load_data(config.map_schema_filename,config.maps_foldername);


from Autoencoder import *;
import tensorflow as tf;
def init_model(input_dim,hidden_node_number,learning_rate):
	

	print "layers:[",input_dim,hidden_node_number,input_dim,"]";	

	print "initializing autoencoder...";
	ae=Autoencoder(input_dim,hidden_node_number);
	print "autoencoder initialized";

	#initialize optimizers

	optimizer_def=tf.train.AdamOptimizer(
			learning_rate);
	optimizer=optimizer_def.minimize(
				loss=ae.cost,
				var_list=[
					ae.encW,
					ae.decW,
					ae.decb,
					ae.encb
					]);
	#end initialize optimizer 

	sess=tf.Session();
	sess.run(tf.initialize_all_variables());
	ae.optimizer=optimizer;
	ae.sess=sess;
	return ae;

def init_model_config(config):
	config.ae=init_model(config.data.input_dim,
				   config.hidden_node_number,config.learning_rate);

def load_model(ae,load_filename):
	
	#loading optimization variables
	try:
		if(load_filename!=""):
			loader=tf.train.Saver(ae.getOptimizationVarsList());
			loader.restore(ae.sess,load_filename+".ckpnt");
			print "variables restored from %s"%(load_filename);
	except NameError:
		print "starting from a new model";

def load_model_config(config):
	return load_model(config.ae,config.load_filename);

def save_model(ae,save_filename):
	#saving only ae vars
	saver=tf.train.Saver(ae.getOptimizationVarsList());
	saved_filename=saver.save(ae.sess,save_filename+".ckpnt");
	print "model saved in %s"%(saved_filename);

def save_model_config(config):
	return save_model(config.ae,config.save_filename);


import TrainingAE as t;
def train(data,ae,start_epoch,epochs,batch_size):
	trainingset,valset,testset=t.train(data.instances,
									ae,ae.optimizer,
									start_epoch,epochs,
									batch_size,
									ae.sess);
	data.trainingset=trainingset;
	data.validationset=valset;
	data.testset=testset;

def train_config(config,start_epoch,epochs):
	return train(config.data,config.ae,
			  start_epoch,epochs,config.batch_size)

import visualization as vis;
def visualize_model(ae,data):
	influence,impact=ae.sess.run(
	fetches=[ae.encW,ae.decW]);
	
	influence=np.transpose(influence);
	vis.plot_maps([influence,impact],data.coords,
			   data.width,data.height);

def visualize_effect(ae,testset,data):
	recon=ae.sess.run(fetches=ae.reconstruction,feed_dict={
		ae.x:testset
		});
	vis.plot_maps([testset,recon],
			   data.coords,data.width,data.height);

class Params:
	def init(self,map_schema_filename,maps_foldername,
						hidden_node_number,learning_rate,batch_size,
						load_filename,save_filename):
		self.map_schema_filename=map_schema_filename;
		self.maps_foldername=maps_foldername;
		self.hidden_node_number=hidden_node_number;
		self.learning_rate=learning_rate;
		self.batch_size=batch_size;
		self.load_filename=load_filename;
		self.save_filename=save_filename;
	def save(self,filename):
		f=open(filename,'w');
		f.write(self.map_schema_filename+"\n");
		f.write(self.maps_foldername+"\n");
		f.write(str(self.hidden_node_number)+"\n");
		f.write(str(self.learning_rate)+"\n");
		f.write(str(self.batch_size)+"\n");
		f.write(self.load_filename+"\n");
		f.write(self.save_filename+"\n");
	def load(self,filename):
		f=open(filename,'r');
		
		self.map_schema_filename=f.readline().strip();
		self.maps_foldername=f.readline().strip();
		self.hidden_node_number=int(f.readline().strip());
		self.learning_rate=float(f.readline().strip());
		self.batch_size=int(f.readline().strip());
		self.load_filename=f.readline().strip();
		self.save_filename=f.readline().strip();