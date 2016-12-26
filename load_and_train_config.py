from Helpers import *;

params=Params();
params.load("tmp.cfg");

load_data_config(params);

init_model_config(params);

load_model_config(params);

train_config(params,0,50);

save_model_config(params);