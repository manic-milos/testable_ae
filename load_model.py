import Helpers;

#initializing the model
#params
hidden_node_number=15;
learning_rate=0.00001;
input_dim=len(instances[0]);

ae,optimizer,sess=Helpers.init_model(input_dim,
									 hidden_node_number,
									 learning_rate);
#loading the model
#params
load_filename="tmp";

Helpers.load_model(ae,sess,load_filename);
