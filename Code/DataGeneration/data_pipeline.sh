#!/bin/bash

#run this pipeline from DataGeneration directory
channel_widths='15.0 14.0'  #12.0 13.0 10.0 11.0  one decimal
viscosity_range='3.5 4 4.5 5 5.5'
#pressure_range='120'

#source ../venv_thesis/bin/activate
echo 'Scenario to run:'
read scenario

#create all meshes --> works
# for width in ${channel_widths}
# do 
# 	python create_mesh.py $scenario $width -nopopup
# 	cp Meshes/channel_${scenario}_w${width}.msh2 openfoam/run/Channel_${scenario}/	
# done

#deactivate

#run simulations, store VTK files and convert VTK to PNG ---> works
#cd openfoam/run/channel_${scenario}
source venv_data_thesis/bin/activate
for width in ${channel_widths}
do
	# #create mesh, set boundary conditions -> remove previous parabolic inletcondition (dependent on diameter)
	# rm -r dynamicCode
	# docker run --name openfoam_container --rm -t -v "$(PWD):/data" -w /data sylleh/openfoam9-macos ./Allrun ${scenario} ${width}
	
	# #run simulation for different viscosities 
	# for vis in ${viscosity_range}
	# do
	# 	docker run --name openfoam_container --rm -t -v "$(PWD):/data" -w /data sylleh/openfoam9-macos ./Simulate ${vis}
	# 	rm -r VTK/allPatches
	# 	mkdir -p ../../../VTK_files/${scenario}/w${width}
	# 	cp -a VTK/ ../../../VTK_files/${scenario}/w${width}/
	# 	cd ../../../VTK_files/${scenario}/w${width}
	# 	for file in data_*; do mv "$file" "${file/data/${scenario}_w${width}_v${vis}}"; done
	# 	cd ../../../openfoam/run/channel_${scenario}
	# 	echo 'simulation for width '${width}' and viscocity '${vis}' is done'
	# done 
	
	
	
	python VTKtoPNG.py $scenario $width

	
done
deactivate
#cd ../../..
echo 'All done'



#for pressure in ${pressure_range}
#do
#	python bloodflow_functions.py $pressure
#	deactivate
#	cp outlet_p.csv openfoam/run/Channel_$scenario
#	cp inlet_v.csv openfoam/run/Channel_$scenario
#done


