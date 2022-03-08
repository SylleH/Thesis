""""
Output OpenFoam to input ML network -->  convert VTK files to PNG images
Author: Sylle Hoogeveen
"""

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
import os

def load_data(filepath):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filepath)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    data = reader.GetOutput()

    # get coordinates of unstructured nodes in mesh and convert to np array
    nodes_vtk_array = data.GetPoints().GetData()
    nodes_numpy_array = vtk_to_numpy(nodes_vtk_array)
    x, y = nodes_numpy_array[:, 0], nodes_numpy_array[:, 1]
    return data, x, y

def plot_convex(filename, data,x,y):

    #get the calculated velocity and pressure on POINTS (can also get data in cells, with GetCellData())
    velocity_vtk_array = data.GetPointData().GetArray('U')
    pressure_vtk_array= data.GetPointData().GetArray('p')
    U = vtk_to_numpy(velocity_vtk_array)
    p = vtk_to_numpy(pressure_vtk_array)
    Ux, Uy = U[:,0], U[:,1]
    U_mag = np.sqrt(np.power(Ux,2) + np.power(Uy,2))

    #create structured grid for plotting
    npts=1000
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)

    xi = np.linspace(xmin,xmax,npts)
    yi = np.linspace(ymin, ymax, npts)
    grid = np.meshgrid(xi,yi)


    Ui = griddata((x,y), U_mag, (xi[None,:], yi[:,None]), method = 'cubic',fill_value=0)

    plt.pcolor(xi,yi,Ui, cmap=cm.jet, vmin=0, vmax=0.8)
    plt.axis('off')
    plt.xlim(left=-0.0025, right =0.0025)
    plt.savefig('Data_generated/Straight/'+filename.split('.')[0]+'.png', bbox_inches='tight')
    plt.colorbar()
    plt.show()


for filename in os.listdir('VTK_files/straight'):
    filepath = 'VTK_files/straight/'+filename
    data, x, y = load_data(filepath)
    plot_convex(filename, data, x, y)

# filepath = 'VTK_files/straight/channel_straight_130.vtk'
# data, x, y = load_data(filepath)
# plot_convex('channel_straight_130.vtk', data, x, y)

def display_grid_points(x,y):

    plt.scatter(x,y)
    plt.show()

def load_velocity(data):
    surface = vtk.vtkDataSetSurfaceFilter()
    surface.SetInputData(data)
    surface.Update()
    surface_data = surface.GetOutput()

    # Extracting triangulation information
    triangles = surface_data.GetPolys()#.GetData()
    #ToDo: extract connectivity from returned vtkCellArray --> this is the cell point id
    #.GetData() returns offset,id1,id2,id3? One entry goes missing
    print(triangles)
    points = surface_data.GetPoints()
    print(points)

    # Mapping data: cell -> point
    mapper = vtk.vtkCellDataToPointData()
    mapper.AddInputData(surface_data)
    mapper.Update()
    mapped_data = mapper.GetOutput()

    # Extracting interpolate point data
    udata = mapped_data.GetPointData().GetArray('U')
    pdata = mapped_data.GetPointData().GetArray('p')

    ntri = int(triangles.GetNumberOfTuples()/4)
    npts = points.GetNumberOfPoints()
    nvls = udata.GetNumberOfTuples()

    tri = np.zeros((ntri, 3))
    x = np.zeros(npts)
    y = np.zeros(npts)
    ux = np.zeros(nvls)
    uy = np.zeros(nvls)

    for i in range(0, ntri):
        tri[i, 0] = triangles.GetTuple(4*i + 1)[0]
        tri[i, 1] = triangles.GetTuple(4*i + 2)[0]
        tri[i, 2] = triangles.GetTuple(4*i + 3)[0]

    for i in range(npts):
        pt = points.GetPoint(i)
        x[i] = pt[0]
        y[i] = pt[1]

    for i in range(0, nvls):
        U = udata.GetTuple(i)
        ux[i] = U[0]
        uy[i] = U[1]

    return (x, y, tri, ux, uy)

# x, y, tri, ux, uy = load_velocity(surface_data)
# plt.tricontour(x, y, tri, ux, 16, linestyles='-',
#                colors='black', linewidths=0.5)
# plt.tricontourf(x, y, tri, ux, 16)



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





