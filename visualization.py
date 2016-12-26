import numpy as np;
import tensorflow as tf;

def mapnodestoimg(ionodes,x,y,coords):
	imgdata=np.zeros([x,y],dtype=np.float32);
	for i in range(len(coords)):
		imgdata[coords[i][0],coords[i][1]]=ionodes[i];
	imgdata=np.transpose(imgdata);
	return imgdata


# %% maps=list(termalmaps)
def plot_maps(maps,coords,width,height):
	import matplotlib.pyplot as plt;
	maps=np.array(maps);
	shape=maps.shape;
	num_vert=1;
	num_hor=len(maps);
	if(len(shape)>2):
		num_vert=shape[0];
		num_hor=shape[1];

	fig, axs=plt.subplots(num_vert,num_hor);
	for j in range(num_vert):
		for i in range(num_hor):
			if(num_vert==1):
				fig.colorbar(
					axs[i].imshow(
						mapnodestoimg(maps[i],width,
					height,coords))
					,ax=axs[i]);
			else:
				fig.colorbar(
					axs[j][i].imshow(
						mapnodestoimg(maps[j][i],width,height,coords))
					,ax=axs[j][i]);
	fig.show();
	plt.draw();
	plt.waitforbuttonpress();

def make_img_summary(nodes,coords,x,y,img_variable,sess):
	imgdata=mapnodestoimg(nodes,x,y,coords);
	sess.run(tf.assign(img_variable,imgdata));

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'