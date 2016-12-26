import Helpers;

#data loading 
#params
map_schema_filename="data/mapschema.csv";
maps_foldername="data/converted";

[instances,
	coords,
	img_dims,
	onlyfiles]=Helpers.load_data(map_schema_filename,maps_foldername);
