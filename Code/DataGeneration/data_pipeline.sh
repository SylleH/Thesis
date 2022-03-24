#!/bin/bash

#run this pipeline from DataGeneration directory
channel_widths='0.75 1.0' #NO SPACES
#pressure_range='120'

source ../venv_thesis/bin/activate
echo 'Scenario to run:'
read scenario

#create all meshes --> works
for width in ${channel_widths}
do 
	python create_mesh.py $scenario $width
	cp Meshes/channel_${scenario}_w${width}.msh2 openfoam/run/Channel_${scenario}/	
done

# deactivate

#run simulations, store VTK files ---> works
for width in ${channel_widths}
do
	cd openfoam/run/channel_${scenario}
	docker run --name openfoam_container --rm -t -v "$(PWD):/data" -w /data sylleh/openfoam9-macos ./Allrun ${scenario} ${width}
	rm -r VTK/allPatches
	cp -a VTK/ ../../../VTK_files/${scenario}/w${width}/
	mkdir -p ../../../VTK_files/${scenario}/w${width}
	cd ../../../VTK_files/${scenario}/w${width}
	for file in data_*; do mv "$file" "${file/data/${scenario}_w${width}_}"; done
	cd ../../..
	
done

# source venv_data_thesis/bin/activate
# python VTKtoPNG.py $scenario $width
# deactivate

echo 'All done'



#for pressure in ${pressure_range}
#do
#	python bloodflow_functions.py $pressure
#	deactivate
#	cp outlet_p.csv openfoam/run/Channel_$scenario
#	cp inlet_v.csv openfoam/run/Channel_$scenario
#done


