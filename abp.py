# -*- co
"""
Created on Wed Jan 16 13:44:19 2019

@author: Sudipta
"""
import os
import math as m
import random as r
import matplotlib.pyplot as plt
import numpy.random
import numpy as np
sigma=1
eps=1
Dr=0.01
dt=0.001;
v0=1;
Dt=0.1
box_length=30

dirname1='../plots'
dirname2='../data'

#the basic idea is that the particle instances are embedded in an arena instance.
#the paricle has x,y attribute corresponding to its coordinate in arena
#and a theta attribute corresponding to its orientation of motion


class particle: 
    def __init__(self,x_cor,y_cor,t):
        self.x=x_cor;
        self.y=y_cor;
        self.theta=t;
    
    def updatex(self,newx):
        self.x=newx;
    
    def updatey(self,newy):
        self.y=newy;
    
    def updatet(self,newt):
        self.theta=newt;
    
    def director(self):
        ar=[0,0];
        ar[0]=m.cos(self.theta);
        ar[1]=m.sin(self.theta);
        return ar;
    
    def coordinate(self):
        ar=[0]*2
        ar[0]=self.x;
        ar[1]=self.y;
        return ar;

class arena:
    def __init__(self,l):
        self.particles=[];
        self.length=l;

    def initialise(self,nos,l):
        occupancy_bucket=[]
        bound=int(m.floor(self.length/l))
        for i in range(1,bound):
            for j in range(1,bound):
                temp=[i,j]
                occupancy_bucket.append(temp)
        r.shuffle(occupancy_bucket)
        for i in range(nos):
            if len(occupancy_bucket)>=1:
                temp=occupancy_bucket.pop()
                x=temp[0]*l
                y=temp[1]*l
                t=r.random()*(2*m.pi)-m.pi
                temp_part=particle(x,y,t)
                self.particles.append(temp_part)
        

    
    def visualise(self,i):
        x_cors=[];
        y_cors=[];
        for entry in self.particles:
            x_cors.append(entry.x);
            y_cors.append(entry.y);
        plt.figure();
        plt.scatter(x_cors,y_cors,facecolors='None',edgecolors='r',s=20)
#        plt.xlim(0,self.length);
#        plt.ylim(0,self.length);
        plt.savefig("%s/plot_at_%d.png"%(dirname1,i))
        plt.clf()
        plt.close('all')
#the force is on particle 1 due to particle 2
#it has been derived by differentiting WCA potential term with respect to coordinates of particles 1
def force(par1,par2):
    delx=par1.x-par2.x
    dely=par1.y-par2.y
    delx=delx-(round(delx/float(box_length)))*box_length
    dely=dely-(round(dely/float(box_length)))*box_length
    r_2=(delx**2) +(dely**2)
    r=r_2**0.5
    if r<(2**(1/6))*sigma:
        t1=2*pow(sigma,12)/pow(r,14);
        t2=pow(sigma,6)/pow(r,8);
        f_x=24*eps*delx*(t1-t2);
        f_y=24*eps*dely*(t1-t2);
        return [f_x,f_y];
    else:
        return [0,0]
#forceij gives out a matrix where (i,j)th entry is a tuple
#representing the force on ith particle by jth particle
def forceij(aren):
    nos=len(aren.particles)
    flist=[]
    for i in range(nos):
        row=[]
        for j in range(nos):
            temp=[0,0]
            row.append(temp)
        flist.append(row)
        
    for i in range(nos):
        for j in range(i,nos):
            if(j==i):
                pass
            else:
                par1=aren.particles[i]
                par2=aren.particles[j]
                temp=force(par1,par2)
                flist[i][j]=temp
                flist[j][i][0]=-1*temp[0]
                flist[j][i][1]=-1*temp[1]
    return flist
#the forcelist gives a linear list with ith entry being the total force on ith particle 
def forcelist(aren):
    nos=len(aren.particles)
    fij=forceij(aren)
    result=[]
    for i in range(nos):
        temp=[0,0]
        for j in range(nos):
            temp1=fij[i][j]
            temp[0]+=temp1[0]
            temp[1]+=temp1[1]
        result.append(temp)
    return result
#delrf gives the displacement of a particle due to force
def delrf(aren):
    temp=forcelist(aren)
    result=[]
    for entry in temp:
        t=[0,0]
        t[0]=dt*entry[0]
        t[1]=dt*entry[1]
        result.append(t)
    return result
#delr_therm gives displacement due to translational diffusion
def delr_therm(aren):
    result=[]
    for entry in aren.particles:
        temp=[0,0]
        temp[0]=m.sqrt(2*Dt*dt)*numpy.random.normal()
        temp[1]=m.sqrt(2*Dt*dt)*numpy.random.normal()
        result.append(temp)
    return result
#delr_dir gives displacement due to orientation of the particle
def delr_dir(aren):
    result=[]
    for entry in aren.particles:
        temp=entry.director()
        t=[0,0]
        t[0]=v0*temp[0]*dt
        t[1]=v0*temp[1]*dt
        result.append(t)
    return result



#updates theta attribute for a particle
def updatetheta(particl):
    add=m.sqrt(2*Dr*dt)*numpy.random.normal()
    t0=particl.theta+add
    particl.updatet(t0)
#updates the whole arena for one time step
def updatearena(aren):
    temp1=delrf(aren)
    temp2=delr_therm(aren)
    temp3=delr_dir(aren)
    for i in range(len(aren.particles)):
        temppart=aren.particles[i]
        tx=temppart.x+temp1[i][0]+temp2[i][0]+temp3[i][0]
        ty=temppart.y+temp1[i][1]+temp2[i][1]+temp3[i][1]
        if(tx<0):
            tx=tx+aren.length
        if(tx>aren.length):
            tx=tx-aren.length
        if(ty<0):
            ty=ty+aren.length
        if(ty>aren.length):
            ty=ty-aren.length
        temppart.updatex(tx)
        temppart.updatey(ty)
        updatetheta(temppart)
        
#updates arena for 'times' number of timesteps
def updatetimes(aren,times):
    for i in range(times):
        updatearena(aren)
        aren.visualise(i)

def updateandstore(aren,times):
    nos=len(aren.particles)
    data=np.zeros((times,2*nos))
    for i in range(times):
        for index in range(nos):
            data[i,2*index]=aren.particles[index].x
            data[i,2*index+1]=aren.particles[index].y
        updatearena(aren)
    return data


def updatestore(aren,times):
    nos=len(aren.particles)
    short_data=np.zeros((1000,2*nos+1))
    time_500=int(float(times)/float(500))
    ld_x=np.zeros((time_500,nos+1))
    ld_y=np.zeros((time_500,nos+1))
    ld_t=np.zeros((time_500,nos+1))
    row_count=0
    for i in range(times):
    
        if i<1000:
            for index in range(nos):
                short_data[i,2*index]=aren.particles[index].x
                short_data[i,2*index+1]=aren.particles[index].y
            short_data[i,2*nos]=i
            filename0='%s/short_data.txt'%dirname2
            np.savetxt(filename0,short_data) 
            
        if((i+1)%500==0):
            for index in range(nos):
                ld_x[row_count,index]=aren.particles[index].x
                ld_y[row_count,index]=aren.particles[index].y
                ld_t[row_count,index]=aren.particles[index].theta
            ld_x[row_count,nos]=i+1
            ld_y[row_count,nos]=i+1
            ld_t[row_count,nos]=i+1
            row_count=row_count+1
            filename1='%s/ld_x.txt'%dirname2
            filename2='%s/ld_y.txt'%dirname2
            filename3='%s/ld_t.txt'%dirname2
            np.savetxt(filename1,ld_x)
            np.savetxt(filename2,ld_y)
            np.savetxt(filename3,ld_t)
            aren.visualise(i)
        updatearena(aren)
    
    
        
    
    

t=arena(box_length)
t.initialise(390,1.4)
updatestore(t,200000)






