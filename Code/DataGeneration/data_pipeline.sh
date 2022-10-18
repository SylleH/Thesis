#!/bin/bash

#run this pipeline from DataGeneration directory
channel_widths='7.0 8.0 9.0 10.0 11.0'  #one decimal 
viscosity_range='3.5 4.0 4.5 5.0 5.5' #3.0 3.5 
outlet_widths='6.3 6.7 7.1'	# 3.25 
#pressure_range='120'

#source ../venv_thesis/bin/activate
source venv_data_thesis/bin/activate

echo 'Scenario to run:'
read scenario

#create all meshes --> works
for width in ${channel_widths}
do 
	for outlet in ${outlet_widths}
	do
		python create_mesh.py $scenario $width $outlet -nopopup
		cp Meshes/channel_${scenario}_w${width}_o${outlet}.msh2 openfoam/run/Channel_${scenario}/
	done	
done

#deactivate

#run simulations, store VTK files and convert VTK to PNG ---> works
cd openfoam/run/channel_${scenario}
for width in ${channel_widths}
do
	for outlet in ${outlet_widths}
	do
		#create mesh, set boundary conditions -> remove previous parabolic inletcondition (dependent on diameter)
		# Note: Allrun doesn't run anything, just sets boundary patches to correct type :-)
		rm -r dynamicCode
		docker run --name openfoam_container --rm -t -v "$(PWD):/data" -w /data sylleh/openfoam9-macos ./Allrun ${scenario} ${width} ${outlet}
		
		# #run simulation for different viscosities 
		for vis in ${viscosity_range}
		do
			docker run --name openfoam_container --rm -t -v "$(PWD):/data" -w /data sylleh/openfoam9-macos ./Simulate ${vis}
			rm -r VTK/allPatches
			mkdir -p ../../../VTK_files/${scenario}/w${width}/o${outlet}/v${vis}
			cp -a VTK/ ../../../VTK_files/${scenario}/w${width}/o${outlet}/v${vis}/
			cd ../../../VTK_files/${scenario}/w${width}/o${outlet}/v${vis}
			for file in data_*; do mv "$file" "${file/data/${scenario}_w${width}_o${outlet}_v${vis}}"; done
			cd ../../../../../openfoam/run/channel_${scenario}
			echo 'simulation for width '${width}', outlet '${outlet}' and viscocity '${vis}' is done'
		done 
	done	
	# cd ../../..
		
	# python VTKtoPNG.py $scenario $width

	
done
deactivate

echo 'All done'



#for pressure in ${pressure_range}
#do
#	python bloodflow_functions.py $pressure
#	deactivate
#	cp outlet_p.csv openfoam/run/Channel_$scenario
#	cp inlet_v.csv openfoam/run/Channel_$scenario
#done


