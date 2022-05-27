#!/usr/bin/python

""""
Output OpenFoam to input ML network -->  convert VTK files to PNG images
Author: Sylle Hoogeveen
"""

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gmsh
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from mpl_toolkits import mplot3d
import matplotlib.tri as tri
import os
import sys

def load_VTK_data(filepath):
    """
    load the data from VTK file, unstructured grid
    :param filepath: to VTK file
    :return x_2d, y_2d: x and y coordinates of data points on plane z=0 in VTK file
    :return U_2d: magnitude of velocity at data points on plane z=0
    """
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filepath)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    data = reader.GetOutput() #recorded data in VTK file --> not sure what all data is, depends on what reader is chosen


    # get coordinates of unstructured nodes in mesh and convert to np array
    nodes_vtk_array = data.GetPoints().GetData()
    nodes_numpy_array = vtk_to_numpy(nodes_vtk_array)
    x, y, z = nodes_numpy_array[:, 0], nodes_numpy_array[:, 1], nodes_numpy_array[:,2]

    # get the calculated velocity and pressure on POINTS (can also get data in cells, with GetCellData())
    velocity_vtk_array = data.GetPointData().GetArray('U')
    pressure_vtk_array= data.GetPointData().GetArray('p')
    U = vtk_to_numpy(velocity_vtk_array)
    p = vtk_to_numpy(pressure_vtk_array)
    Ux, Uy = U[:,0], U[:,1]
    U_mag = np.sqrt(np.power(Ux,2) + np.power(Uy,2))

    # only use the nodes for which z=0, so that are on the original surface
    id = []
    for i in range(len(z)):
        if z[i] == 0:
            id.append(i)

    x_2d = np.zeros(len(id))
    y_2d = np.zeros(len(id))
    U_2d = np.zeros(len(id))
    for j in range(len(id)):
        x_2d[j] = x[id[j]]
        y_2d[j] = y[id[j]]
        U_2d[j] = U_mag[id[j]]


    return x_2d, y_2d, U_2d

def load_GMSH_data(filepath):
    """
    Load mesh data. WARNING: make sure to load mesh on which OpenFoam simulation was run
    :param filepath: path to .msh2 file
    :return x, y: x and y coordinates of nodes on SURFACE 1 (plane for which z=0)
    :return triangle_id: np arrray with vertices of each triangle on SURFACE 1, converted to id matching position in coordinate vectors
    """

    gmsh.initialize()
    gmsh.open(filepath)

    # Get mesh nodes from entity (2, 1) = the surface
    # nodeCoords is (x,y,z) coordinate corresponding to node index nodeTag
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes(2,1)

    nodes_id_coords = np.zeros((len(nodeTags),3))
    for i in range(len(nodeTags)):
        nodes_id_coords[i,0] = nodeTags[i]
        nodes_id_coords[i,1] = nodeCoords[3*i]      # x coordinate
        nodes_id_coords[i,2] = nodeCoords[3*i+1]    # y coordinate
        # note we ditch the z coordinate

    x = nodes_id_coords[:,1]
    y = nodes_id_coords[:,2]

    # Get mesh elements from the entity (2 ,1) = the surface
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(2,1)

    #create matrix containing all node id's for the triangles
    triangle_id = np.zeros((len(elemTags[0]),3))
    for j in range(len((elemTags)[0])):
        triangle_id[j,0] = elemNodeTags[0][3*j]     #vertice 1
        triangle_id[j,1] = elemNodeTags[0][3*j+1]   #vertice 2
        triangle_id[j,2]= elemNodeTags[0][3*j+2]    #vertice 3


    #convert triangle id's to index in nodes_id_coords instead of original tag number
    triangle_id = np.array(triangle_id)
    for i in range(len(nodeTags)):
        idx = np.where(triangle_id == nodeTags[i])
        for j in range(len(idx[0])):
            triangle_id[idx[0][j], idx[1][j]] = i
    triangle_id = triangle_id.astype(int)

    gmsh.clear()
    gmsh.finalize()

    return x, y, triangle_id


def plot_and_save(scenario, filename, x,y, U, triangle_id):
    """
    Function to create and save png image from VTK and GMSH data
    :param scenario: either straight, bifurcation, bend or branch
    :param filename: given as scenario_timestep, used to store png file
    :param x, y: x and y coordinates of mesh points (from GMSH or VTK data, should be equal)
    :param triangle_id: array containing the index of vertices of each triangle, corresponds to position in x,y arrays
    """

    #U_mid is average value of three vertices, len(U_mid) = len(ElemTags), value for each triangle
    #U_mid= U[triangle_id].mean(axis=1)


    #create triangulation and set interpolator, cubic gives visually best result
    triang = tri.Triangulation(x,y,triangle_id)
    interpolator = tri.CubicTriInterpolator(triang, U)
    #interpolator = tri.LinearTriInterpolator(triang, U)

    #create structured grid for plotting
    npts=1000
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    xi = np.linspace(xmin,xmax,npts)
    yi = np.linspace(ymin, ymax, npts)
    xg,yg = np.meshgrid(xi,yi)

    #interpolate U on new grid, using triangulation
    U_int = interpolator(xg,yg)


    #plot and save figure
    plt.figure()
    plt.pcolor(xg,yg, U_int, cmap=cm.gray, vmin=0, vmax=0.5)
    plt.axis('off')
    plt.xlim(left=-0.03, right =0.03)
    plt.savefig('Data_generated/test_case/' + os.path.splitext(filename)[0] + '.png', bbox_inches='tight')
    #plt.savefig('Data_generated/'+scenario+'/'+os.path.splitext(filename)[0]+'.png', bbox_inches='tight')
    #plt.colorbar()
    #plt.show()
    plt.close()


def main(argv):
    scenario = argv[0]
    width = argv[1]

    mesh_path = 'Meshes/channel_'+scenario+'_w'+width+'.msh2'

    x, y, triangle_id = load_GMSH_data(mesh_path)
    count = 0
    for filename in os.listdir('VTK_files/'+scenario+'/w'+width+''):
        if filename == '.DS_Store':
            continue
        VTK_path = 'VTK_files/'+scenario+"/w"+width+'/'+filename
        x_VTK, y_VTK, U = load_VTK_data(VTK_path)

        # check if mesh and VTK file are matched correctly, only need to check once
        if count == 0:
            if np.allclose(x_VTK, x, atol=1e-05) == False:
                sys.exit('x coordinates mesh and VTK data are not equal')
            if np.allclose(y_VTK, y, atol=1e05) == False:
                sys.exit('y coordinates mesh and VTK data are not equal')
            else:
                print('Mesh and VTK data coordinates match, yeah')

        plot_and_save(scenario, filename, x, y, U, triangle_id)
        count+=1


if __name__ == "__main__":
    matplotlib.use('agg')
    main(sys.argv[1:])

"""
These are help functions and old attempt, please ignore
"""
def display_grid_points(x,y):
    """
    Function to show mesh points from VTK data
    :param x: x coordinate of mesh points
    :param y: y coordinate of mesh points
    """
    plt.scatter(x,y)
    plt.show()


def plot_convex(filename, x,y, U):
    """
    Function to create and save png image from VTK data
    :param filename: given as scenario_timestep, used to store png file
    :param data: VTK data from simulation
    :param x: x coordinate of mesh points
    :param y: y coordinate of mesh points
    """

    #create structured grid for plotting
    npts=1000
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)

    xi = np.linspace(xmin,xmax,npts)
    yi = np.linspace(ymin, ymax, npts)
    grid = np.meshgrid(xi,yi)


    Ui = griddata((x,y), U, (xi[None,:], yi[:,None]), method = 'cubic',fill_value=0)

    plt.pcolor(xi,yi,Ui, cmap=cm.jet, vmin=0, vmax=0.8)
    plt.axis('off')
    plt.xlim(left=-0.0025, right =0.0025)
    plt.savefig('Data_generated/Straight/'+filename.split('.')[0]+'.png', bbox_inches='tight')
    plt.colorbar()
    plt.show()


def create_mask(x,y):
    THRESHOLD = 0.001

    tree = cKDTree(np.c_[x,y])
    dist, _ = tree.query(np.c_[xi, yi], k=1)
    dist = dist.reshape(xi.shape)
    m1 = (dist>THRESHOLD)
    m2 = np.any([dist>THRESHOLD , dist>-THRESHOLD])

    xp,yp= xi[m1],yi[m1]
    #xp,yp = xp[0],yp[0]
    zp = np.nan + np.zeros_like(xp)
    return xp, yp, zp

# xp,yp,zp = create_mask(x,y)
# U_mask = griddata((np.r_[x,xp], np.r_[y,yp]), np.r_[U_mag, zp], (xi[None,:], yi[:,None]), method = 'linear',fill_value=0)





