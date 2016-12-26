

from os import listdir, makedirs
from os.path import isfile, join, exists

import sys

class Dims(object):
	x=0;
	y=0;
	def __init__(self,x,y):
		self.x=x;
		self.y=y;
		

def convert(from_path,to_path=None,schema_name=None,debug=False):
	mypath=from_path;
	files=[f for f in listdir(mypath) if isfile(join(mypath,f))]
	#create schema
	filename=files[0];
	f=open(mypath+"/"+filename,"r");
	if(schema_name==None):
		schema_path=raw_input("new schema name:");
	else:
		schema_path=schema_name;
	w=open(schema_path,"w");
	width=0;
	height=0;
	line=f.readline();
	coords=[];
	while(line !=""):
		cells=line.split(' ');
		width=0;
		for cell in cells:
			if(cell!='NA'):
				try:
					floatvalue=float(cell);
					coords.append((width,height));
				except ValueError:
					if(debug):
						print("not float:",cell);
					continue;
			width+=1;
		height+=1;
		line=f.readline();
	print len(coords);
	print "width="+str(width);
	print "height="+str(height);
	w.write(str(width)+" "+str(height)+"\n");

	for item in coords:
		w.write(str(item[0])+" "+str(item[1]));
		w.write("\n");

	w.close();
	if(to_path is None):
		converted_path=raw_input("converted maps directory name:");
	else:
		converted_path=to_path;
	if not exists(converted_path):
			makedirs(converted_path)
	#convert files
	fileindex=0;
	for filename in files:
		digits=2
		print "{0}{1:{2}}%".format(
						"\b" * (digits + 1+1), 
						int((fileindex+0.0)/(len(files))*100),
						digits),
		fileindex=fileindex+1;
		sys.stdout.flush()
		f=open(mypath+"/"+filename,"r");
		w=open(converted_path+"/"+filename,"w");
		for line in f:
			cells=line.split(' ');
			for cell in cells:
				if(cell!='NA'):
					try:
						value=float(cell);
						w.write(cell+" ");
					except ValueError:
						continue;
		f.close();
		w.close();

def revert(from_path,schemapath,to_path=None,debug=False):
	files=[f for f in listdir(from_path) if isfile(join(from_path,f))];
	
	
	f=open(schemapath,'r');
	coords=[];
	xy=f.readline().split(' ');
	img_dims=Dims(int(xy[0]),int(xy[1]));
	line=f.readline();
	while(line!=""):
		xy=line.split(' ');
		try:
			coords.append((int(xy[0]),int(xy[1])));
		except ValueError:
			continue;
		line=f.readline();
	if(debug):
		print("coords len:",len(coords));
	for mapfile in files:
		f=open(from_path+"/"+mapfile,"r");
		
		wf=open(to_path+"/"+mapfile,'w');
		x=0;
		y=0;
		ind=0;
		for line in f:
			cells=line.split(' ');
			if(debug):
				print("cells len:",len(cells));
			for cell in cells:
				if(cell==""):
					if(debug):
						print("empty cell");
					continue;
				#print(coords[ind]);
				tox=coords[ind][0];
				toy=coords[ind][1];
				if(toy>y):
					y=y+1;
					if(img_dims.x-x>1):
						for i in range(img_dims.x-x-1):
							wf.write("NA ");
						wf.write("NA");
					elif(img_dims.x-x==1):
						wf.write("NA");
					x=0;
					wf.write("\n");
					if(debug):
						print("newline");
				if(tox>x):
					for i in range(tox-x):
						wf.write("NA ");
				
				wf.write(cell);
				if(tox<img_dims.x-1):
					wf.write(" ");
				ind=ind+1;
				x=tox+1;
				#raw_input();
		if(x<img_dims.x):
			if(img_dims.x-x==1):
				wf.write("NA");
			else:
				for i in range(img_dims.x-x-1):
					wf.write("NA ");
				wf.write("NA");
			wf.write("\n");
		f.close();
		wf.close();
		