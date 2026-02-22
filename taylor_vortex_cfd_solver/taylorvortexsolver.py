# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 23:50:33 2022

@author: Ioannis Skounakis-Kounavis
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcol
import matplotlib.cm as cm
import pandas as pd
from collections import namedtuple
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable #anchor colorbar
cbarcolors=["#0207fa","#04dede","#91f01d","#ffff00","#fc1605","#820e04"]
cm1 = mcol.LinearSegmentedColormap.from_list("newmap",cbarcolors,N=256,gamma=1.5)
plt.rcParams['text.usetex'] = True
################################
##GENERAL FUNCTIONS##
def distance(x1,y1,x2,y2): #distance between 2D points, accepts 4 coords
################################
    a,b=np.array([x1,y1],dtype=object),np.array([x2,y2],dtype=object)
    return np.linalg.norm(a-b)
def showgrid(x,y): #plots the lines of the grid - debug
    global ax
    X,Y=np.meshgrid(x, y)
    plt.plot(X,Y, color='k',linewidth=".5")
    plt.plot(np.transpose(X),np.transpose(Y), color='k',linewidth=".5")
##########Initial ω0-ψ0#########
def W0(x, y): #i
################################
    global x1,y1,x2,y2
    r1=distance(x,y,x1,y1)
    r2=distance(x,y,x2,y2)
    w1=(2-r1**2)*np.exp((1-r1**2)/2)
    w2=-(2-r2**2)*np.exp((1-r2**2)/2)
    return w1+w2
def psy0(x, y):
    global x1,y1,x2,y2
    r1=distance(x,y,x1,y1)
    r2=distance(x,y,x2,y2)
    psi1=np.exp((1-r1**2)/2)
    psi2=-np.exp((1-r2**2)/2)
    return psi1+psi2
##GRID-MESH FUNCTIONS##
################################
def num2coord(n, flag=False): #one number to ij or xy
    global Nx,rangex,rangey,dx,dy
    if type(n)==list:
        ii=rangex[0]+(n[0]-1)*dx
        jj=rangey[0]+(n[1]-1)*dy
        return [ii,jj]
    else:
        ii=n%Nx
        if ii==0:
            ii=Nx
        jj=(n-ii)/Nx+1
        if flag:
            ii=rangex[0]+(ii-1)*dx
            jj=rangey[0]+(jj-1)*dy
    return [ii,jj]
################################
def deriv(ii,jj,dgrid): #derivative
    global dx,dy
    #derivative to x
    try:
        dfdx=(dgrid[ii+1][jj]-dgrid[ii-1][jj])/(2*dx)
    except KeyError:
        try:
            dfdx=(dgrid[ii+1][jj]-dgrid[ii][jj])/(dx) #forward
        except KeyError:
            dfdx=(dgrid[ii][jj]-dgrid[ii-1][jj])/(dx) #backward
    #derivative to y
    try:
        dfdy=(dgrid[ii][jj+1]-dgrid[ii][jj-1])/(2*dy)
    except KeyError:
        try:
            dfdy=(dgrid[ii][jj+1]-dgrid[ii][jj])/(dy) #forward
        except KeyError:
            dfdy=(dgrid[ii][jj]-dgrid[ii][jj-1])/(dy) #backward
    return dfdx,dfdy
################################
def ar2mesh(ar,A=False): #array to pandas mesh
    global Nx,Ny
    if A:
        return pd.DataFrame(ar,index=np.arange(1, Nx*Ny+1),columns=np.arange(1,Nx*Ny+1))
    return pd.DataFrame(ar,index=np.arange(Ny,0,-1),columns=np.arange(1,Nx+1))
################################
def buildb(W): #add ω mesh values in b
    global Nx,Ny
    b=np.zeros(Nx*Ny)
    for j in range(2,Ny):
        for i in range(2,Nx):
            k = (j-1)*Nx+i-1
            b[k]=-W[i][j]
    return b
################################    
def boundmat(b,w,A): #add boundary sums in b
    global Nx, Ny, ddx, ddy, dx, dy, x, y
    for ii in range(Nx*Ny):
        if not(-2*ddx-2*ddy == A[ii][ii]):
            X, Y = np.meshgrid(x, y)
            bX,bY=np.full((Ny-2,Nx-2),num2coord(ii+1,True)[0]),np.full((Ny-2,Nx-2),num2coord(ii+1,True)[1])
            distances=np.sqrt((bX-X[1:-1,1:-1])**2+(bY-Y[1:-1,1:-1])**2)
            b[ii]=-np.sum((1/(2*np.pi))*np.log(distances)*w[1:-1,1:-1]*dx*dy)
    return b
##############################
def animateVORT(frames): #matplotlib animation 
    global Wdt, psy, A, dt, cbarticks,x,y
    Wdt,psy=updateMESH(Wdt,psy,False,True)
#################################################
    ax.clear()
    CS=ax.contourf(x,y,Wdt,cbarticks,cmap=cm1,vmin=-4, vmax=4)
    ax.contour(x,y,Wdt, cbarticks, colors='black',linewidths=.5,vmin=-4, vmax=4)
    ax.axis("equal")
    cbar  = fig.colorbar(CS, cax=cax, ticks=cbarticks)
    ax.set_title("Vorticity Field, t = "+str(round(frames*dt,2))+"s",fontsize = 'small')
#################################################
def updateMESH(w,psy,plotting=True,viscosity=False):
    global na, dt, A, timestep,cm1,cbarticks
##################################################
    W=ar2mesh(w[::-1])
    ################################      
    #Find u,v mesh, dω/dx dω/dy and eventually ω(x,y,t+Δt) 
    U=np.gradient(psy,dy,dx)[0] #u=dψ/dy, v=-dψ/dx
    V=-np.gradient(psy,dy,dx)[1]
    Wx=np.gradient(w,dy,dx)[1]
    Wy=np.gradient(w,dy,dx)[0]
    Wxx=np.gradient(Wx,dy,dx)[1]
    Wyy=np.gradient(Wy,dy,dx)[0]
##################################################
    Wdt=w-dt*(U*Wx+V*Wy)+(na*(Wxx+Wyy) if viscosity else 0)
##################################################
    W=ar2mesh(Wdt[::-1])
    ################################       
    #Boundary Conditions & right hand matrix (b)
    b=buildb(W) #b is the 1*NxNy matrix of results
    b=boundmat(b,w,A)
    ################################         
    #Solving Linear System Ax=b where x = [ψ1...ψNxNy]
    solution = np.linalg.solve(A,b)
    psy=solution.reshape(Ny,Nx)
##########################

    if plotting:
        plt.figure(dpi=500)
        CS=plt.contourf(x,y,Wdt,cbarticks,cmap=cm1,vmin=-4, vmax=4)
        plt.contour(x,y,Wdt, cbarticks, colors='black',linewidths=.5,vmin=-4, vmax=4)
        plt.colorbar(CS,ticks=cbarticks)
        #showgrid(x,y)
        plt.axis('equal')
        plt.title("Vorticity Field, t = "+str(round(timestep*dt,2))+"s",fontsize = 'small')
        plt.show()

##########################
    return Wdt,psy


##MESH INITIALIZATION##
na = 1.5e-4
x1,x2=0,0
y1,y2=1.5,-1.5 #Vortex Coordinates
rangex=[-5,5]
rangey=[-5,5]
Nx,Ny=100,100
dx,dy=((rangex[1]-rangex[0])/(Nx-1)),((rangey[1]-rangey[0])/(Ny-1))
ddx,ddy=1/(dx**2),1/(dy**2)
x = np.linspace(rangex[0],rangex[1], Nx)
y = np.linspace(rangey[0],rangey[1], Ny)
X, Y = np.meshgrid(x, y, sparse=True)
################################
##set up animation plotsn##
fig, ax = plt.subplots(dpi=400)
div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')
cbarticks=np.arange(-4,5,1)
dt=0.01 #time step
##W0##
w0 = W0(X,Y) #Initial Conditions for ω
W=ar2mesh(w0[::-1])
##SOLUTION MATRIX NxNy*NxNy dimensions##
#initialize matrices
A=np.identity(Nx*Ny) #A is the NxNy*NxNy matrix of coefficients
for j in range(2,Ny):
    for i in range(2,Nx):
        k = (j-1)*Nx+i-1 #numbering starts from 0
        A[k][k-(Nx)],A[k][k-1],A[k][k],A[k][k+1],A[k][k+(Nx)]=ddy,ddx,-2*ddx-2*ddy,ddx,ddy 
###############################
#first iteration
b=buildb(W)
b=boundmat(b, w0, A)
solution = np.linalg.solve(A,b)
psy=solution.reshape(Ny,Nx)
#error calculations
psyerror = (psy-psy0(X,Y))/max(solution)
Wdt=w0


U=np.gradient(psy,dy,dx)[0] #u=dψ/dy, v=-dψ/dx
V=-np.gradient(psy,dy,dx)[1]
UV=np.sqrt(U**2+V**2)
#################################
#Initial Mesh - Analytical Values - Plots Only
#################################

fig1,ax1=plt.subplots(1,2,dpi=700,figsize=(9, 4.5), tight_layout=True)
fig1.suptitle(r"Analytical Vorticity \& Stream Function")
wcontour=ax1[0].contourf(x,y,Wdt,10,cmap=cm1,vmin=-4, vmax=4)
ax1[0].contour(x,y,Wdt, 10, colors='black',linewidths=.5,vmin=-4, vmax=4)
ax1[0].axis("equal")
cbar  = fig1.colorbar(wcontour, ax=ax1[0])
ax1[0].set_title(r'Vorticity Field, $t = 0s$',fontsize = 'medium')
ax1[0].set_xlabel(r'$x$ axis')
ax1[0].set_ylabel(r'$y$ axis')

wcontour=ax1[1].contourf(x,y,psy0(X,Y),10,cmap=cm1)
ax1[1].contour(x,y,psy0(X,Y), 10, colors='black',linewidths=.5)
ax1[1].axis("equal")
cbar  = fig1.colorbar(wcontour, ax=ax1[1])
ax1[1].set_title(r'Stream function, $t = 0s$',fontsize = 'medium')
ax1[1].set_xlabel(r'$x$ axis')
ax1[1].set_ylabel(r'$y$ axis')
#######################################################
fig2,ax2=plt.subplots(dpi=700)
wcontour=ax2.contourf(x,y,psyerror,10,cmap=cm1)
ax2.contour(x,y,psyerror, 10, colors='black',linewidths=.5)
ax2.axis("equal")
cbar = fig2.colorbar(wcontour, ax=ax2)
ax2.set_title(r'Absolute Error',fontsize = 'medium')
ax2.set_xlabel(r'$x$ axis')
ax2.set_ylabel(r'$y$ axis')
#######################################################
figc,axc=plt.subplots(1,2, dpi=700,figsize=(9, 4.5), tight_layout=True)
figc.suptitle(r"Analytical vs Numerical Stream Function")
wcontour=axc[0].contourf(x,y,psy0(X,Y),10,cmap=cm1)
axc[0].contour(x,y,psy0(X,Y), 10, colors='black',linewidths=.5)
axc[0].axis("equal")
cbar  = fig1.colorbar(wcontour, ax=axc[0])
axc[0].set_title(r'Analytical',fontsize = 'medium')
axc[0].set_xlabel(r'$x$ axis')
axc[0].set_ylabel(r'$y$ axis')

wcontour=axc[1].contourf(x,y,psy,10,cmap=cm1)
axc[1].contour(x,y,psy, 10, colors='black',linewidths=.5)
axc[1].axis("equal")
cbar  = fig1.colorbar(wcontour, ax=axc[1])
axc[1].set_title(r'Numerical',fontsize = 'medium')
axc[1].set_xlabel(r'$x$ axis')
axc[1].set_ylabel(r'$y$ axis')
#######################################################
fig1,ax1=plt.subplots(1,3,dpi=700,figsize=(9, 3), tight_layout=True)
fig1.suptitle(r"Horizontal \& Vertical Velocity Fields")
wcontour=ax1[0].contourf(x,y,U,10,cmap=cm1)
ax1[0].contour(x,y,U, 10, colors='black',linewidths=.5)
ax1[0].axis("equal")
cbar  = fig1.colorbar(wcontour, ax=ax1[0])
ax1[0].set_title(r'Horizontal Velocity, u',fontsize = 'medium')
ax1[0].set_xlabel(r'$x$ axis')
ax1[0].set_ylabel(r'$y$ axis')

wcontour=ax1[1].contourf(x,y,V,10,cmap=cm1)
ax1[1].contour(x,y,V, 10, colors='black',linewidths=.5)
ax1[1].axis("equal")
cbar  = fig1.colorbar(wcontour, ax=ax1[1])
ax1[1].set_title(r'Vertical  Velocity, v',fontsize = 'medium')
ax1[1].set_xlabel(r'$x$ axis')
ax1[1].set_ylabel(r'$y$ axis')

wcontour=ax1[2].contourf(x,y,UV,10,cmap=cm1)
ax1[2].contour(x,y,UV, 10, colors='black',linewidths=.5)
ax1[2].axis("equal")
cbar  = fig1.colorbar(wcontour, ax=ax1[2])
ax1[2].set_title(r'Velocity Vector',fontsize = 'medium')
ax1[2].set_xlabel(r'$x$ axis')
ax1[2].set_ylabel(r'$y$ axis')
#######################################################

plt.figure(dpi=700)
ax2 = plt.axes(projection='3d')
ax2.grid(False)
ax2.elev=20
ax2.azim=-50
ax2.plot_surface(X, Y, psy, rstride=1, cstride=1,cmap=cm1, edgecolor='none')
#plotting directly


'''
for timestep in range(100):
    Wdt,psy=updateMESH(Wdt,psy,True,False)
'''

#animating

#ani = animation.FuncAnimation(fig, animateVORT, int(round(10/dt,0)), blit=False, interval=1000*dt)
#ani.save("trial.mp4")  





