import images_to_nodes as itn;
import filecmp;
from os import listdir;
from os.path import isfile,join, exists;
test_list=["conversion_test_data/data"];

results_folder="conversion_test_data/results";
comparison_folder="conversion_test_data/reverted";
schema_path="conversion_test_data/schema.csv";

	
#TODO cleanup before


for item in test_list:
	itn.convert(item,results_folder,schema_path,True);
	itn.revert(results_folder,schema_path,
			comparison_folder,True)
	fromfiles=[f for f in listdir(item) if isfile(join(item,f))];
	tofiles=[f for f in listdir(comparison_folder) if isfile(join(comparison_folder,f))];
	for file1 in fromfiles:
		i=0;
		while(tofiles[i]!=file1):
			i=i+1;
		print(filecmp.cmp(join(item,file1),
					join(comparison_folder,tofiles[i])));
#TODO cleanup after