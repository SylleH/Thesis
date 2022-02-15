cd $FOAM_RUN
cp -r $FOAM_TUTORIALS/incompressible/simpleFoam/pitzDaily .
cd pitzDaily
blockMesh
simpleFoam
paraFoam
exit
exit
cd channel_branch
ls
cd $FOAM_RUN/channel_branch
gmshToFoam(channel_branch.msh)
gmshToFoam channel_branch.msh
gmshToFoam channel_branch.msh2
gmshToFoam channel_branch.msh2
gmshToFoam channel_branch.msh
gmshToFoam channel_branch.msh2
gmshToFoam channel_branch.msh2
exit
cd $FOAM_RUN/channel_branch
gmshToFoam channel_branch.msh2
gmshToFoam channel_branch.msh2
exit
cd $FOAM_RUN/channel_branch
gmshToFoam channel_branch.msh2
fluentMeshToFoam channel_branch.msh
gmshToFoam channel_branch.msh2
foamInfo -a symmetryPlane
foamInfo -a symmetryPlane
ls
icoFoam
icoFoam
icoFoam
icoFoam
icoFoam
icoFoam
paraview
paraview
icoFoam
paraFoam
exit
cd run/channel_branch
icoFoam
paraFoam
ok
exit
cd $FOAM_RUN/pitzDaily
paraFoam
exit
cd run/pitzdaily
paraFoam
exit
cd run/channel_branch
foamToVTK -allPatches
icoFoam
icoFoam
icoFoam
icoFoam
icoFoam
find $FOAM_SRC/finiteVolume/fields/fvPatchFields -type f -name "*.H" |\\ xargs grep -1 Function1 | xargs dirname | sort
find $FOAM_SRC/finiteVolume/fields/fvPatchFields -type f -name "*.H" | xargs grep -1 Function1 | xargs dirname | sort
icoFoam
icoFoam
icoFoam
icoFoam
icoFoam
icoFoam
icoFoam
icoFoam
foamToVTK
foamToVTK -allpatches
foamToVTK -allPatches
icoFoam
foamToVTK -allPatches
foamToVTK -allPatches
icoFoam
icoFoam
icoFoam
foamToVTK -allPatches
icoFoam
icoFoam
icoFoam
icoFoam
icoFoam
icoFoam
icoFoam
icoFoam
icoFoam
foamToVTK -allPatches
icoFoam
icoFoam
icoFoam
foamToVTK -allPatches
exit
cd run/channel_bend90
gmshToFoam channel_bend90.msh2
icoFoam
foamToVTK -allPatches
exit
