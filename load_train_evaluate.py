import TrainingAE as t;
import data_loading as dl;
from Autoencoder import *;
import tensorflow as tf;
import Helpers;

#data loading 
[instances,
	coords,
	img_dims,
	onlyfiles]=Helpers.load_data(map_schema_filename="data/mapschema.csv"
							  ,maps_foldername="data/converted");

#initializing the model
ae,optimizer,sess=Helpers.init_model(input_dim=len(instances[0]),
									 hidden_node_number=15,
									 learning_rate=0.001);
#loading the model
Helpers.load_model(ae,sess,load_filename="tmp");

#training 
[trainingset,
 validationset,
 testset]=t.train(instances,
		ae,
		optimizer,
		start_epoch=0,n_epochs=50,
		batch_size=100,
		sess=sess);

#saving the model
Helpers.save_model(ae,sess,save_filename="tmp");


