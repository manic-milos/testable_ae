import data_loading as dl;
import tensorflow as tf;
import numpy as np;

class Config:
	def __init__(self,load_filename):
		#print "loading filename=",load_filename;
		#inicijalne vrednosti za tensorflow variable
		learning_rate_init=0.00001;
		batch_size_init=25;
		map_schema_filename="mapschemahighres.csv";
		maps_foldername="highres";
		hidden_node_number=8;
		
		self.learning_rate_var=tf.Variable(learning_rate_init,
			name="learning_rate",
			dtype=tf.float32);
		self.batch_size_var=tf.Variable(batch_size_init,name="batch_size");
		self.start_epoch_var=tf.Variable(0,name="start_epoch");
		self.hidden_node_var=tf.Variable(hidden_node_number,
			name="hidden_node_number");
		self.map_schema_var=tf.Variable(map_schema_filename,
			dtype=tf.string,
			name="map_schema");
		self.maps_folder_var=tf.Variable(maps_foldername,
			dtype=tf.string,
			name="map_folder");
		self.sess=tf.Session();
		self.sess.run(tf.initialize_all_variables());
		if(load_filename!=""):
			self.loadParams(load_filename);
		else:
			self.inputParams();
			
	def loadParams(self,load_filename):

		loader=tf.train.Saver(var_list=[
			self.learning_rate_var,
			self.batch_size_var,
			self.hidden_node_var,
			self.start_epoch_var,
			self.map_schema_var,
			self.maps_folder_var]);
		loader.restore(self.sess,(load_filename+".ckpnt"));
		
		self.updateParams();
		
	def inputParams(self):

		maps_foldername=raw_input("folder containing the maps:");
		self.setMapFoldername(maps_foldername);
		map_schema_filename=raw_input("filename for map_schema:");
		self.setMapSchemaFoldername(map_schema_filename);
		hidden_node_number=input("number of nodes in hidden layer:");
		self.setHiddenNodeNumber(hidden_node_number);
		learning_rate_init=input("initial learning rate:");
		self.setLearningRate(learning_rate_init);
		batch_size_init=input("batch size:");
		self.setBatchSize(batch_size_init);
		self.setStartEpoch(0);
		self.updateParams();
		#assign
		
	def setMapFoldername(self,maps_foldername):
		self.sess.run(tf.assign(self.maps_folder_var,maps_foldername));
		
	def setMapSchemaFoldername(self,map_schema_filename):
		self.sess.run(tf.assign(self.map_schema_var,map_schema_filename));
		
	def setHiddenNodeNumber(self,hidden_node_number):
		self.sess.run(tf.assign(self.hidden_node_var,hidden_node_number));
		
	def setLearningRate(self,learning_rate):
		self.sess.run(tf.assign(self.learning_rate_var,learning_rate));
		
	def setBatchSize(self,batch_size):
		self.sess.run(tf.assign(self.batch_size_var,batch_size));
		
	def setStartEpoch(self,start_epoch):
		self.sess.run(tf.assign(self.start_epoch_var,start_epoch));
		
	def updateParams(self):
		
		self.learning_rate=self.sess.run(self.learning_rate_var);
		self.batch_size=self.sess.run(self.batch_size_var);
		self.hidden_node_number=self.sess.run(self.hidden_node_var);
		self.map_schema_filename=self.sess.run(self.map_schema_var);
		self.maps_foldername=self.sess.run(self.maps_folder_var);
		self.start_epoch=self.sess.run(self.start_epoch_var);
		
	def changeParams(self):
		change=input("learning rate:");
		if(change!=''):
			self.setLearningRate(change);
		change=input("batch size:");
		if(change!=''):
			self.setBatchSize(change);
		change=raw_input("map schema:");
		if(change!=''):
			self.setMapSchemaFoldername(change);
		change=raw_input("maps folder:");
		if(change!=''):
			self.setMapFoldername(change);
		self.updateParams();
		
	def printParams(self):
		self.updateParams();
		print "learning rate:",self.learning_rate;
		print "batch_size:",self.batch_size;
		print "epochs trained:",self.start_epoch;
		print "map schema filename:",self.map_schema_filename;
		print "maps folder:", self.maps_foldername;
		
	def setAdaptiveLearningRate(self,factor):
		self.adapt_learning_rate=tf.assign(self.learning_rate_var,self.learning_rate_var*factor);
		
	def adaptLearningRate(self):
		return self.sess.run(self.adapt_learning_rate);
		
	def reloadParams(self):
		self.setStartEpoch(self.start_epoch);
		self.setHiddenNodeNumber(self.hidden_node_number);
		self.setBatchSize(self.batch_size);
		self.setMapFoldername(self.maps_foldername);
		self.setMapSchemaFoldername(self.map_schema_filename);
		self.setLearningRate(self.learning_rate);
		
		