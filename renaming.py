import data_loading as dl;
import upscaling_ae_def as aedef;
import visualization as vis;
import tensorflow as tf;
import numpy as np;
import math;

load_filename=raw_input("filename:");
print "loading from file:",load_filename;
map_schema_filename="mapschemahighres.csv";
highres_maps_folder="highres";
hidden_node_number=8;
map_schema_var=tf.Variable(map_schema_filename,
	dtype=tf.string,
	name="map_schema");
hidden_node_var=tf.Variable(hidden_node_number,
	name="hidden_node_number");
highres_folder_var=tf.Variable(highres_maps_folder,
	name="map_folder");
sess=tf.Session();
sess.run(tf.initialize_all_variables());
loader=tf.train.Saver(var_list=[
			map_schema_var,
			hidden_node_var,
			highres_folder_var]);
loader.restore(sess,load_filename);
map_schema_filename=sess.run(map_schema_var);
highres_maps_folder=sess.run(highres_folder_var);
hidden_node_number=sess.run(hidden_node_var);
print "checkpoint restored";
print "map schema:",map_schema_filename;
print "map folder:",highres_maps_folder;
print "hidden nodes:",hidden_node_number;

save_filename=raw_input("saving filename:");
print "save file:",save_filename;

[coords,img_dims]=dl.load_schema(map_schema_filename);
input_dim=len(coords);
representation=hidden_node_number;
print "initializing loading vars";
high_decW=tf.Variable(
		initial_value=tf.random_normal(
			[representation,input_dim],
			-math.sqrt(6.0/(input_dim+representation)),
			math.sqrt(6.0/(input_dim+representation))),
		dtype=tf.float32,
		name='newDecW');
high_encW=tf.Variable(
	initial_value=tf.random_normal(
		[input_dim, representation],
		-math.sqrt(6.0/(input_dim+representation)),
		math.sqrt(6.0/(input_dim+representation))),
	name='newEncW');
high_encb=tf.Variable(tf.zeros([representation]),
	name='newEncb');
high_decb=tf.Variable(
		tf.zeros([input_dim]),
		name='newDecb');

sess.run(tf.initialize_all_variables());
loader=tf.train.Saver(var_list=[
	high_decW,
	high_encW,
	high_encb,
	high_decb,
	])
print "restoring";
loader.restore(sess,load_filename);
vis.plot_maps(sess.run(high_decW),coords,
	img_dims['x'],img_dims['y']);
rundecW=sess.run(high_decW);
runencW=sess.run(high_encW);
runencb=sess.run(high_encb);
rundecb=sess.run(high_decb);
print "initializing ae...";
ae=aedef.autoencoder_contd(input_dim,hidden_node_number);
print "ae initialized";
print "initializing parameters:";

learning_rate_init=input("initial learning rate:");

batch_size_init=input("batch size");

learning_rate=tf.Variable(learning_rate_init,
	name="learning_rate",
	dtype=tf.float32);
batch_size=tf.Variable(batch_size_init,name="batch_size");
start_epoch=tf.Variable(0,name="start_epoch");

sess.run(tf.initialize_all_variables());

print "assigning";
sess.run(tf.assign(map_schema_var,map_schema_filename));
sess.run(tf.assign(hidden_node_var,hidden_node_number));
sess.run(tf.assign(highres_folder_var,highres_maps_folder));

sess.run(tf.assign(ae['encW'],runencW));
sess.run(tf.assign(ae['encb'],runencb));
sess.run(tf.assign(ae['decW'],rundecW));
sess.run(tf.assign(ae['decb'],rundecb));
vis.plot_maps(sess.run(ae['decW']),coords,
	img_dims['x'],img_dims['y']);
vis.plot_maps(sess.run(high_decW),coords,
	img_dims['x'],img_dims['y']);
print "saving"

saver=tf.train.Saver(var_list=[
	ae['encW'],
	ae['decW'],
	ae['encb'],
	ae['decb'],
	learning_rate,
	batch_size,
	start_epoch,
	map_schema_var,
	hidden_node_var,
	highres_folder_var]);
saver.save(sess,save_filename);