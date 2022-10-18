#!/bin/bash

#run this pipeline from DataGeneration directory
widths=(11.0)  #one decimal 
viscosities=(5.0) #3.0 3.5 
outlets=(6.25)	# 3.25 

size1=${#widths[@]}
size2=${#outlets[@]}
size3=${#viscosities[@]}

#Create some random inlet patterns

source venv_data_thesis/bin/activate
for value in {1}
do
	index1=$(($RANDOM % $size1))
	index2=$(($RANDOM % $size2))
	index3=$(($RANDOM % $size3))
	echo ${widths[$index1]}
	echo ${outlets[$index2]}
	echo ${viscosities[$index3]}

	vel_pars=($(python random_inlet_vel.py))

	Umax=0.216 #${vel_pars[0]}
	Umin=0.006 #${vel_pars[1]}
	Utwo=0.12 #${vel_pars[2]}
	tb=0.23 #${vel_pars[3]}
	td=0.29 #${vel_pars[4]}
	tp=0.16 #${vel_pars[5]}

	echo $Umax 
	echo $Umin 
	echo $Utwo
	echo $tb 
	echo $td
	echo $tp

	cd openfoam/run/channel_bifurcation

	rm -r dynamicCode
	docker run --name openfoam_container --rm -t -v "$(PWD):/data" -w /data sylleh/openfoam9-macos ./Allrun bifurcation ${widths[$index1]} ${outlets[$index2]} $Umax $Umin $Utwo $tb $td $tp
	docker run --name openfoam_container --rm -t -v "$(PWD):/data" -w /data sylleh/openfoam9-macos ./Simulate ${viscosities[$index3]}
	rm -r VTK/allPatches
	mkdir -p ../../../VTK_files/bifurcation/newdata/w${widths[$index1]}/o${outlets[$index2]}/v${viscosities[$index3]}/vp6
	cp -a VTK/ ../../../VTK_files/bifurcation/newdata/w${widths[$index1]}/o${outlets[$index2]}/v${viscosities[$index3]}/vp6/
	cd ../../../VTK_files/bifurcation/newdata/w${widths[$index1]}/o${outlets[$index2]}/v${viscosities[$index3]}/vp6
	for file in data_*; do mv "$file" "${file/data/bifurcation_w${widths[$index1]}_o${outlets[$index2]}_v${viscosities[$index3]}_vp6}"; done
	cd ../../../../../../../
	echo 'simulation for width '${widths[$index1]}', outlet '${outlets[$index2]}' and viscocity '${viscosities[$index3]}' is done'
done
deactivate

echo 'All done'




