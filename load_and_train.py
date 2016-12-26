
import Helpers;

#data loading 
#params
map_schema_filename="data/mapschema.csv";
maps_foldername="data/converted";

data=Helpers.load_data(map_schema_filename,maps_foldername);

#initializing the model
#params
hidden_node_number=15;
learning_rate=0.00001;
input_dim=len(data.instances[0]);

ae=Helpers.init_model(input_dim,
							hidden_node_number,
							learning_rate);
#loading the model
#params
load_filename="tmp";

Helpers.load_model(ae,load_filename);

#training 
#params
batch_size=100;
start_epoch=0;end_epoch=50;

Helpers.train(data,
		ae,
		start_epoch,end_epoch,
		batch_size);

#saving the model
#params
save_filename="tmp";

Helpers.save_model(ae,save_filename);


