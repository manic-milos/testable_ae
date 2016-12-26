from os import listdir, makedirs
from os.path import isfile, join, exists

import sys

def makeTable(nodes_dir,resulting_file_path):
	files=[f for f in listdir(nodes_dir) if isfile(join(nodes_dir,f))];
	w=open(resulting_file_path,"w");
	for entry in files:
		f=open(join(nodes_dir,entry),"r");
		cells=[];
		for line in f:
			w.write(line);
			w.write("\n");
		f.close();
	w.close();