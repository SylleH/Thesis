#!/usr/bin/python

"""
Author: Sylle Hoogeveen
Functions to create multiple meshes with GMSH for OpenFoam simulations
"""

import gmsh
import sys
import math
import numpy as np

def build_channel_straight(width, Lc):
    gmsh.model.add("channel_straight")
    factory = gmsh.model.geo
    mm = 1e-03

    w = width*mm #0.75*mm
    h = 100*mm #156*mm #mean length descending thoratic aorta is 332mm

    factory.addPoint(-w,0,0, Lc,1)
    factory.addPoint(-w,h,0, Lc,2)
    factory.addPoint(w,h,0, Lc,3)
    factory.addPoint(w,0,0, Lc,4)

    factory.addLine(1, 2, 1)
    factory.addLine(2, 3, 2)
    factory.addLine(3, 4, 3)
    factory.addLine(4, 1, 4)

    factory.addCurveLoop([1,2,3,4],5)
    factory.addPlaneSurface([5], 1)

    ov = factory.copy([(2, 1)])
    ov2 = factory.extrude([(2, 1)], 0, 0, Lc, numElements=[1], recombine=True)

    factory.synchronize()

    gmsh.model.addPhysicalGroup(2, [19], 1)
    gmsh.model.addPhysicalGroup(2, [1], 2)
    gmsh.model.addPhysicalGroup(2, [32], 3)
    gmsh.model.addPhysicalGroup(2, [27], 4)
    gmsh.model.addPhysicalGroup(2, [23], 5)
    gmsh.model.addPhysicalGroup(2, [31], 6)
    gmsh.model.addPhysicalGroup(2, [88], 8)
    gmsh.model.addPhysicalGroup(3, [1], 2)

    gmsh.model.setPhysicalName(2, 1, "left")
    gmsh.model.setPhysicalName(2, 2, "back")
    gmsh.model.setPhysicalName(2, 3, "front")
    gmsh.model.setPhysicalName(2, 4, "right")
    gmsh.model.setPhysicalName(2, 5, "top")
    gmsh.model.setPhysicalName(2, 6, "bottom")
    gmsh.model.setPhysicalName(3, 2, "the volume")

    gmsh.model.mesh.generate(3)
    name='channel_straight_w'+str(width)
    gmsh.write('Meshes/'+name+".msh2")

def build_channel_branch(width, Lc):
    gmsh.model.add("channel_branch")
    factory = gmsh.model.geo
    mm = 1e-03

    w = width * mm
    r = 2*w
    h1 = 100*mm
    h2 = 80*mm
    y = math.sqrt(math.pow(w,2) +2*w*r)

    factory.addPoint(-w, 0, 0, Lc, 1)
    factory.addPoint(-w, h1, 0, Lc, 2)
    factory.addPoint(w, h1, 0, Lc, 3)
    factory.addPoint(w, h2, 0, Lc,4)
    factory.addPoint(w+r, h2, 0, Lc, 5) #midpoint circle
    factory.addPoint(w+r, h2-r, 0 , Lc, 6)
    factory.addPoint(w+r, h2-r-w, 0 , Lc, 7)
    factory.addPoint(w, h2-y, 0 , Lc, 8)
    factory.addPoint(w, 0, 0, Lc, 9)

    factory.addLine(1, 2, 1)
    factory.addLine(2, 3, 2)
    factory.addLine(3, 4, 3)
    factory.addCircleArc(4,5,6,4)
    factory.addLine(6,7,5)
    factory.addCircleArc(7,5,8,6)
    factory.addLine(8,9,7)
    factory.addLine(9,1,8)

    factory.addCurveLoop([1,2,3,4,5,6,7,8], 9)
    factory.addPlaneSurface([9],1)

    ov = factory.copy([(2, 1)])
    ov2 = factory.extrude([(2, 1)], 0, 0, Lc, numElements=[1], recombine=True)

    factory.synchronize()
    gmsh.model.addPhysicalGroup(2, [60], 1)
    gmsh.model.addPhysicalGroup(2, [1], 2)
    gmsh.model.addPhysicalGroup(2, [39,43,55,51], 3)
    gmsh.model.addPhysicalGroup(2, [47], 4)
    gmsh.model.addPhysicalGroup(2, [31], 5)
    gmsh.model.addPhysicalGroup(2, [59], 6)
    gmsh.model.addPhysicalGroup(2, [35], 7)
    gmsh.model.addPhysicalGroup(3, [1], 2)

    gmsh.model.setPhysicalName(2, 1, "front")
    gmsh.model.setPhysicalName(2, 2, "back")
    gmsh.model.setPhysicalName(2, 3, "right_main")
    gmsh.model.setPhysicalName(2, 4, "right_branch")
    gmsh.model.setPhysicalName(2, 5, "left")
    gmsh.model.setPhysicalName(2, 6, "bottom")
    gmsh.model.setPhysicalName(2, 6, "top")
    gmsh.model.setPhysicalName(3, 2, "the volume")

    gmsh.model.mesh.generate(3)

    name='channel_branch_w'+str(width)
    gmsh.write('Meshes/'+name+".msh2")


def build_channel_bend(width, deg, Lc):
    factory = gmsh.model.geo
    mm = 1e-03

    gmsh.model.add("channel_bend")

    w = width*mm
    angle= np.deg2rad(deg) # 0< deg < 180
    r1 = 12*mm
    r2 = r1+w


    factory.addPoint(0, 0, 0, Lc, 1) #midpoint circle
    factory.addPoint(-r1, 0, 0, Lc, 2)
    factory.addPoint(-r2, 0, 0, Lc, 3)
    factory.addPoint(-r2 * math.cos(angle), -r2 * math.sin(angle), 0, Lc, 4)
    factory.addPoint(-r1*math.cos(angle), -r1*math.sin(angle), 0, Lc, 5)

    factory.addLine(2, 3, 1)
    factory.addCircleArc(3, 1, 4, 2)
    factory.addLine(4, 5, 3)
    factory.addCircleArc(5, 1, 2, 4)

    factory.addCurveLoop([1,2,3,4], 5)
    factory.addPlaneSurface([5],1)

    ov = factory.copy([(2, 1)])
    ov2 = factory.extrude([(2, 1)], 0, 0, Lc, numElements=[1], recombine=True)

    factory.synchronize()
    gmsh.model.addPhysicalGroup(2, [32], 1)
    gmsh.model.addPhysicalGroup(2, [1], 2)
    gmsh.model.addPhysicalGroup(2, [23], 3)
    gmsh.model.addPhysicalGroup(2, [27], 4)
    gmsh.model.addPhysicalGroup(2, [31], 5)
    gmsh.model.addPhysicalGroup(2, [19], 6)
    gmsh.model.addPhysicalGroup(3, [1], 2)

    gmsh.model.setPhysicalName(2, 1, "front")
    gmsh.model.setPhysicalName(2, 2, "back")
    gmsh.model.setPhysicalName(2, 3, "outer_arc")
    gmsh.model.setPhysicalName(2, 4, "right")
    gmsh.model.setPhysicalName(2, 5, "inner_arc")
    gmsh.model.setPhysicalName(2, 6, "top")
    gmsh.model.setPhysicalName(3, 2, "the volume")

    gmsh.model.mesh.generate(3)

    name='channel_bend_w'+str(width)+'_a'+str(deg)
    gmsh.write('Meshes/'+name+".msh2")

def build_channel_bifurcation(width, Lc):
    gmsh.model.add("channel_bifurcation")
    factory = gmsh.model.geo
    mm = 1e-03

    width2 = 6*mm

    e1 = 12*mm #1.25*mm e1 > e3
    e2 = e1 + width2
    e3 = width*mm
    h1 = 45*mm
    h2 = 55*mm
    h3 = 25*mm
    r1 = 10*mm
    r2 = 5*mm

    def hypot(a, b):
        return math.sqrt(a * a + b * b)


    ccos = (-h3 * r2 + e1 * hypot(h3, hypot(e1, r2))) / (h3 * h3 + e1 * e1)
    ssin = math.sqrt(1 - ccos * ccos)



    factory.addPoint(-e1, 0, 0, Lc, 1)
    factory.addPoint(-e1 - e2, 0, 0, Lc, 2)
    factory.addPoint(-e1, h1, 0, Lc, 3)
    factory.addPoint(-e3-r1, h1 + math.sqrt(math.pow(r1,2)-math.pow((e3+r1-e1),2)), 0, Lc, 4) #middlepoint left circle
    factory.addPoint(-e3, h1 + math.sqrt(math.pow(r1,2)-math.pow((e3+r1-e1),2)), 0, Lc, 5)
    factory.addPoint(-e3, h1 + h2, 0, Lc, 6)

    factory.addPoint(e3, h1 + h2, 0, Lc, 7)
    factory.addPoint(e3, h1 + math.sqrt(math.pow(r1,2)-math.pow((e3+r1-e1),2)), 0, Lc, 8)
    factory.addPoint(e3+r1, h1 + math.sqrt(math.pow(r1,2)-math.pow((e3+r1-e1),2)), 0, Lc, 9) #middlepoint right circle
    factory.addPoint(e1, h1 , 0, Lc, 10)
    factory.addPoint(e1 + e2, 0, 0, Lc, 11)
    factory.addPoint(e1, 0, 0, Lc, 12)

    factory.addPoint(r2 / ssin, h3 + r2 * ccos, 0, Lc, 13)
    factory.addPoint(0, h3, 0, Lc, 14) #middlepoint middle circle
    factory.addPoint(-r2 / ssin, h3 + r2 * ccos, 0, Lc, 15)


    factory.addLine(1, 2, 1)
    factory.addLine(2, 3, 2)
    factory.addCircleArc(3, 4, 5, 3)
    factory.addLine(5,6,4)
    factory.addLine(6,7,5)
    factory.addLine(7,8,6)
    factory.addCircleArc(8, 9, 10, 7)
    factory.addLine(10,11,8)
    factory.addLine(11,12,9)
    factory.addLine(12,13,10)
    factory.addCircleArc(13, 14, 15, 11)
    factory.addLine(15,1,12)

    factory.addCurveLoop([1,2,3,4,5,6,7,8,9,10,11,12],13)
    factory.addPlaneSurface([13],1)

    ov = factory.copy([(2, 1)])
    ov2 = factory.extrude([(2,1)], 0,0,Lc,numElements=[1],recombine=True)

    factory.synchronize()
    #gmsh.model.addPhysicalGroup(2,[1,43,47,51,55,59,63,67,71,75,79,83,87,88] , 1)
    gmsh.model.addPhysicalGroup(2, [1], 2)
    gmsh.model.addPhysicalGroup(2, [47, 51, 55], 3)
    gmsh.model.addPhysicalGroup(2, [59], 4)
    gmsh.model.addPhysicalGroup(2, [63, 67, 71], 5)
    gmsh.model.addPhysicalGroup(2,[43, 75], 6)
    gmsh.model.addPhysicalGroup(2,[79,83,87], 7)
    gmsh.model.addPhysicalGroup(2,[88], 8)
    gmsh.model.addPhysicalGroup(3, [1], 2)

    #gmsh.model.setPhysicalName(2, 1, "surfaces")
    gmsh.model.setPhysicalName(2, 2, "front")
    gmsh.model.setPhysicalName(2, 3, "left")
    gmsh.model.setPhysicalName(2, 4, "top")
    gmsh.model.setPhysicalName(2, 5, "right")
    gmsh.model.setPhysicalName(2, 6, "bottom")
    gmsh.model.setPhysicalName(2, 7, "inner_arc")
    gmsh.model.setPhysicalName(2, 8, "back")
    gmsh.model.setPhysicalName(3, 2, "the volume")



    gmsh.model.mesh.generate(3)
    name='channel_bifurcation_w'+str(width)
    gmsh.write('Meshes/'+name+".msh2")


def main(argv):
    gmsh.initialize()

    scenario = argv[0]
    width = float(argv[1])  #width*2 is actual channel width, due to symmetrical building
    deg = 90                #ToDo: make optional system argument, also think about other optional arguments (width branch or bifurcation)
    Lc = 0.001             #this determines the coarseness of the mesh

    if scenario == 'straight':
        build_channel_straight(width, Lc)
    if scenario == 'bend':
        build_channel_bend(width,deg, Lc)
    if scenario == 'branch':
        build_channel_branch(width, Lc)
    if scenario == 'bifurcation':
        build_channel_bifurcation(width, Lc)

    # Launch the GUI to see the results:
    if '-nopopup' not in argv:
        gmsh.fltk.run()

    gmsh.finalize

if __name__ == "__main__":
    main(sys.argv[1:])







