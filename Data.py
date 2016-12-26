class Data:
	def __init__(self,instances,coords,img_dims):
		self.instances=instances;
		self.coords=coords;
		self.img_dims=img_dims;
		self.width=img_dims['x'];
		self.height=img_dims['y'];
		self.input_dim=len(coords);