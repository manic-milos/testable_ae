from Autoencoder import *;
import data_loading as dl;
import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plt;
import visualization as vis;
import math;
import model_evaluation as me;
import time;
import config_load;


#unstable in future versions

flags=tf.app.flags;

flags.DEFINE_string('load', '', 'Filename of a previous save');
flags.DEFINE_boolean('change', False, 'Change loaded params');
flags.DEFINE_string('save', 'tmp', 'Filename to save progress');
flags.DEFINE_integer('epochs',-1, 'Number of epochs to train');



#configuration load
load_filename=flags.FLAGS.load;
print "loading from file:",load_filename;
config=config_load.Config(load_filename);

#end configuration load


#change parameters one by one
change=flags.FLAGS.change;
if(change==True):
	print "Change params:";
	config.changeParams();
#end parameter change


#set factor for adaptive learning rate
config.setAdaptiveLearningRate(0.9);
#end set factor for adaptive learning rate

#schema loading
print "loading schema..."
[coords,img_dims]=dl.load_schema(config.map_schema_filename);
print "schema loaded";
#end schema loading


print "loaded parameters:";
print "layers:[",len(coords),config.hidden_node_number,len(coords),"]";

print "configuration parameters:"
config.printParams();




save_filename=flags.FLAGS.save;
print "saving filename=",save_filename;

n_epochs=flags.FLAGS.epochs;
if(n_epochs<0):
	n_epochs=input("number of epochs to train: ");
print "number_of_epochs",n_epochs;


print "initializing autoencoder...";
input_dim=len(coords);
ae=Autoencoder(input_dim,config.hidden_node_number);
print "autoencoder initialized";

print "loading maps..."
[instances,
	coords,
	img_dims,
	onlyfiles]=dl.load_nodes_with_schema(config.maps_foldername,
	config.map_schema_filename);
print "loading maps completed";


print "initializing sets...";
trainingset=instances[1:int(0.8*len(instances))];
np.random.seed(3);
trainingset=np.random.permutation(np.array(trainingset));
validationset=instances[int(0.8*len(instances)):int(0.9*len(instances))];
testset_beggining=int(0.9*len(instances))
testset=instances[testset_beggining:];
print "sets initialized...";


#initialize optimizer
optimizer_def=tf.train.AdamOptimizer(
	config.learning_rate_var);
optimizer=optimizer_def.minimize(
		loss=ae.cost,
		var_list=[
			ae.encW,
			ae.decW,
			ae.decb,
			ae.encb
			]);
#end initialize optimizer 

sess=config.sess;
sess.run(tf.initialize_all_variables());

loader=tf.train.Saver();
if(load_filename!=""):
	loader.restore(sess,load_filename+".ckpnt");
config.reloadParams();



print "configuration parameters:";
config.printParams();


saver=tf.train.Saver();

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
	

summary_writer=tf.train.SummaryWriter("./"+save_filename+"summary");
training_error_summary=tf.scalar_summary("training_error",
	ae.cost);
validation_error_summary=tf.scalar_summary("validation_error",
	ae.cost);
test_error_summary=tf.scalar_summary("test_error",
	ae.cost);

c=0;
error=10000000000;


img_var=tf.Variable(
	tf.zeros([config.hidden_node_number,img_dims['y'],img_dims['x']]),
	name="img_var");
img_var_resh=tf.reshape(
	img_var,
	[config.hidden_node_number,img_dims['y'],img_dims['x'],1]);
img_sum=tf.image_summary("decoder_image",img_var_resh);
	
sess.run(tf.initialize_variables(var_list=[img_var]));
dec=sess.run(ae.decW);
decimg=[];
for i in range(config.hidden_node_number):
	decimg.append(vis.mapnodestoimg(
		dec[i],
		img_dims['x'],img_dims['y'],
		coords));
sess.run(tf.assign(img_var,np.array(decimg)));
img_sumi=sess.run(img_sum);
summary_writer.add_summary(img_sumi,0);

#reference error

#pix_mean=sess.run(tf.reduce_mean(ae['x']),feed_dict={ae['x']:trainingset});
#print "pixel average:",pix_mean;
#print "reference error:",
#print sess.run(tf.reduce_mean(tf.abs(ae['x']-pix_mean)),
	#feed_dict={ae['x']:testset});
#print "max model error per pixel(training):", sess.run(ae['maxerr'],
	#feed_dict={ae['x']:trainingset});
#print "max model error per pixel(validation):", sess.run(ae['maxerr'],
	#feed_dict={ae['x']:validationset});
#print "max model error per pixel(test):", sess.run(ae['maxerr'],
	#feed_dict={ae['x']:testset});
	
#end reference error


#training
start_time = time.time();

for epoch_i in range(
	config.start_epoch, config.start_epoch+n_epochs):
	i_batch=0;
	max=0;
	print ("epoch "+str(epoch_i)+":\t"),
	for batch_i in range(len(trainingset) // config.batch_size):
		batch_xs= trainingset[i_batch:i_batch+config.batch_size];
		i_batch=i_batch+config.batch_size;
		o,c=sess.run(fetches=[optimizer,ae.cost],
			feed_dict={ae.x:batch_xs});
		if(c>max):
			max=c;
	c,cpp,tesum=sess.run(fetches=[ae.cost,ae.loss_per_pixel,
		training_error_summary],
		feed_dict={
		ae.x:trainingset
		});
	summary_writer.add_summary(tesum,epoch_i);
	cppv,vesum=sess.run(fetches=[ae.loss_per_pixel,
		validation_error_summary],feed_dict={
		ae.x:validationset
		});
	summary_writer.add_summary(vesum,epoch_i);
	
	print "%2.6f\t%2.10f\t%2.10f"%(
		c,cpp,cppv);
	if(c>error*1.01):
		print "new lr",config.adaptLearningRate();
	if(c<error):
		error=c;
	if(epoch_i%100==0 and epoch_i!=config.start_epoch):
		config.setStartEpoch(epoch_i);
		saved_filename=saver.save(sess,save_filename+".ckpnt");
		print "tmp model saved in %s"%(saved_filename)
		print "test set error per pixel:",
		teppx,tes=sess.run([ae.loss_per_pixel,test_error_summary],
			feed_dict={
			ae.x:testset
			});
		print teppx;
		summary_writer.add_summary(tes,epoch_i);
		
#update epoch
config.setStartEpoch(config.start_epoch+n_epochs);

#save model
saved_filename=saver.save(sess,save_filename+".ckpnt");
print "model saved in %s"%(saved_filename)

#calculate time elapsed
print "time elapsed:",time.time()-start_time;

#errors
print "test set error per pixel:",
print ae.getCostPerPixel(sess,testset);
# print "correlation:",me.correlation(testset,sess.run(ae['z'],feed_dict={ae['x']:testset}));
# print "closest representation correlation",
# print me.closestRepCorrelation(testset,sess.run(ae['z'],feed_dict={ae['x']:testset}));
# print "compression loss:",me.compression_loss(testset,sess.run(ae['y'],feed_dict={ae['x']:testset}))
# print "index test:",me.indexTest(testset,sess.run(ae['z'],feed_dict={ae['x']:testset}));
[emax,emin,
cost,recon]=sess.run(fetches=[
	ae.maxerr,ae.minerr,
	ae.cost,ae.reconstruction],
	feed_dict={ae.x:testset});

influence,impact=sess.run(
	fetches=[ae.encW,ae.decW]);

influence=np.transpose(influence);
vis.plot_maps([testset[0:5],recon[0:5]],coords,img_dims['x'],img_dims['y']);
vis.plot_maps([influence,impact],coords,img_dims['x'],img_dims['y']);

print emax,emin;
imin=-1;
imax=-1;
tmin=10000000000000;
tmax=0;

for i in range(len(testset)):
	#costi=sess.run(fetches=ae.loss_per_pixel,
		#feed_dict={ae.x:[testset[i]]});
	costi=ae.getCostPerPixel(sess,[testset[i]]);
	# print costi;
	if(costi>tmax):
		imax=i;
		# print "imax:",costi;
		tmax=costi;
	if(costi<tmin):
		imin=i;
		# print "imin",costi;
		tmin=costi;
		
		
#for i in range(250,len(recon)):	
	#vis.plot_maps(
		#[testset[i],recon[i]],
		#coords,img_dims['x'],img_dims['y']);
# vis.plot_maps(impact,coords,img_dims['x'],img_dims['y']);
