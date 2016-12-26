from osgeo import gdal
import gc;
from os import listdir
from os.path import isfile, join

mypath=raw_input("tif directory path:");
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
saving_path=raw_input("csv saving path:");
fileindex=0;
for name in onlyfiles:
	path=mypath+"/"+name;
	dataset=gdal.Open(path,gdal.GA_ReadOnly);
	band=dataset.GetRasterBand(1);
	array=band.ReadAsArray();
	digits=2;
	print name;
	fileindex+=1;
	writingpath=saving_path+"/"+name+".csv";
	f=open(writingpath,"w");
	for row in array:
		for cell in row:
			if(cell>-10000):
				f.write(str(cell));
				f.write(" ");
			else:
				f.write("NA ");
		f.write("\n");
	f.close();