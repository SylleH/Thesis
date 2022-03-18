#!/bin/bash

channel_widths = 0.75
#pressure_range= ('120')


cd /Documents/'Master Applied Mathematics'/Thesis/Code
source venv_thesis/bin/activate
cd DataGeneration
echo 'Scenario to run:'
read scenario

for width in ${channel_widths}
do 
	python create_mesh.py $scenario $width
	deactivate
	cp Meshes/channel_$scenario.msh2 openfoam/run/Channel_$scenario
	#cd openfoam/run
	
	#docker run -ti --rm -v "$(PWD):/data" -w /data sylleh/openfoam9-macos
	#cd channel_$scenario
	#gmshToFoam channel_$scenario.msh2
	#cd constant
	#now we need to change the boundary dict file somehow 
	#icoFoam
	#foamToVTK -allPatches
	#exit the docker container
	
	#rename the VTK files logically: channel width+BC+timestep in name
	#source venv_data_thesis/bin/activate
	#cp VTK files to VTK_files/$scenario
	#python VTKtoPNG.py $scenario $width
	
done



#for pressure in ${pressure_range}
#do
#	python bloodflow_functions.py $pressure
#	deactivate
#	cp outlet_p.csv openfoam/run/Channel_$scenario
#	cp inlet_v.csv openfoam/run/Channel_$scenario
#done


