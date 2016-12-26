import data_loading as dl;
import Autoencoder as aedef;
import visualization as vis;
import tensorflow as tf;
import numpy as np;

load_filename=raw_input("filename:");
print "load file:",load_filename;

#restoring map schema and hidden node number
map_schema_var=tf.Variable("mapschema.csv",
	dtype=tf.string,
	name="map_schema");
hidden_node_var=tf.Variable(8,
	name="hidden_node_number");
loader=tf.train.Saver(var_list=[
	map_schema_var,
	hidden_node_var]);
sess=tf.Session();
sess.run(tf.initialize_all_variables());
loader.restore(sess,load_filename);
print "checkpoint restored...";


map_schema_filename=sess.run(map_schema_var);
print "map schema:",map_schema_filename;
hidden_node_number=sess.run(hidden_node_var);
print "hidden node number:",hidden_node_number;

#low res map schema
print "loading schemas";
[lcoords,limg_dims]=dl.load_schema(map_schema_filename);
print "low res schema loaded";

#highres map schema
map_schema_highres=raw_input("highres map schema filename:");
[hcoords,himg_dims]=dl.load_schema(map_schema_highres);
print "highres schema loaded";

#getting highres map folder for later use
highres_maps_folder=raw_input("highres maps folder:");

#initializing low res autoencoder
print "initializing ae";
linput_dim=len(lcoords);
hinput_dim=len(hcoords);
lae=aedef.Autoencoder(linput_dim,hidden_node_number,'low_');


print "loading values";
sess.run(tf.initialize_all_variables());
loader=tf.train.Saver({'encW':lae.encW,
			'decW':lae.decW,
			'encb':lae.encb,
			'decb':lae.decb});
loader.restore(sess,load_filename);



print "plotting";
#vis.plot_maps(sess.run(lae.decW),
	#lcoords,limg_dims['x'],limg_dims['y']);
#vis.plot_maps(sess.run(tf.transpose(lae.encW)),
	#lcoords,limg_dims['x'],limg_dims['y']);



print "upscaling"

#initializing highres autoencoder -------CHANGE!
#highae=aedef.autoencoder(hinput_dim,
	#hidden_node_number,
	#sess.run(lae.encW),
	#sess.run(lae.encb),
	#sess.run(lae.decW),
	#sess.run(lae.decb),
	#lcoords,hcoords,
	#'newEncW','newEncb','newDecW','newDecb');

hae=lae.upscale(lcoords,hcoords,sess);
print hae;

#preparing for saving values of node number and map schema
highres_folder_var=tf.Variable(highres_maps_folder,
	name="map_folder");

sess.run(tf.initialize_all_variables());

sess.run(tf.assign(hidden_node_var,hidden_node_number));
sess.run(tf.assign(map_schema_var,map_schema_highres));



print "plotting";

vis.plot_maps(sess.run(hae.encW),
	hcoords,himg_dims['x'],himg_dims['y']);
vis.plot_maps(sess.run(tf.transpose(hae.decW)),
	hcoords,himg_dims['x'],himg_dims['y']);


saver=tf.train.Saver(var_list=[hae.encW,
			hae.decW,
			hae.encb,
			hae.decb,
			map_schema_var,
			hidden_node_var,
			highres_folder_var]);
save_filename=raw_input("saving filename:");
print "saving in %s"%(save_filename);
saver.save(sess,save_filename);