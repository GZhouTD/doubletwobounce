# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 13:57:19 2018

@author: GQZhou
"""

import os
import sys
import numpy as np
import scipy.io as sio
def sciconst():
    scinum = {}
    scinum['e-charge'] = 1.602176565e-19
    scinum['h-plank'] = 6.62607004e-34
    scinum['c-speed'] = 299792458
    return scinum
def ipseed(y,x):
    prime_list = [2]
    v=2
    Jg=True
    while Jg:
        for i in prime_list:
            if v % i == 0:
                 break
            elif i==prime_list[-1]:
                prime_list.append(v)
        v=v+1
        if len(prime_list)==x:
            Jg=False
    prime_list.insert(0,1)
    return prime_list[y:x]
def ReadGenesis2(fto):
    prm1='    z'
    ftc=open(fto,'r')
    alllines=ftc.readlines()
    ftc.close()
    n=-1
    data=[]
    dct={}
    for line in alllines:
        n += 1
        if 'zsep' in line:
            tmp = line.split('=')[-1]
            tmp1 = tmp.replace('D','E')
            zsep = float(tmp1)            # Read zsep
        if 'xlamds' in line:
            tmp = line.split('=')[-1]
            tmp1 = tmp.replace('D','E')
            xlamds = float(tmp1)        # Read xlamds
        if prm1 in line:
            nn=n
        if 'output: slice' in line:
            break
    dct['xlamds']=xlamds
    dct['zsep']=zsep
    data=alllines[nn+1:n-1]
    m=len(data)
    data=''.join(data)
    data=data.split()
    data=np.array(data)
    data=data.reshape(m,3)
    del alllines[0:n-1]
    title=alllines[6]
    stitle=title.split()
    p=len(stitle)
    nslice=int(len(alllines)/(m+7))
    dct['nslice']=nslice
    I = []
    for i in range(0,nslice):
        tmp =alllines[m*i+3].split()
        I.append(float(tmp[0]))
        del alllines[m*i:m*i+7]
    I = np.array(I)
    dct['current'] = I
    data1=''.join(alllines)
    del alllines
    data1=data1.split()
    data1=np.array(data1)
    if len(data1)!=m*nslice*p:
        print(len(data1),m,nslice,p)
        print('total data numer != nslice*column*row')
        sys.exit()
    data1=data1.reshape(m*nslice,p)
    dct['z']=data[:,0]
    dct['aw']=data[:,1]
    dct['qf']=data[:,2]
    for ii in range(0,len(stitle)):
        dct[stitle[ii]]=data1[:,ii].reshape(nslice,m)
    p = dct['power'].astype(np.float)
    phi = dct['phi_mid'].astype(np.float)
    pz = np.mean(p,axis=0)
    ps = p[:,-1]
    phis = phi[:,-1]
    return ps, phis, pz, dct['xlamds'], dct['nslice'], dct['zsep']
def Spectrumf(ps,phis,xlamds,nslice, zsep):
    e_charge = 1.602176565e-19
    h_plank = 6.62607004e-34
    c_speed = 299792458
    f0 = c_speed/xlamds
    deltaT = xlamds*zsep/c_speed
    Fs = 1/deltaT                              # Sampling frequency
    Er = 0.01                                  # Energy resolution(ev)
    Fr = e_charge*Er/h_plank                                # Er = h*Fr/e
    Sp = np.round(Fs/Fr)                       # The estimation of sampling point
    if nslice <= Sp:
        P = np.zeros(int(np.round((Sp-nslice)/2)))
        power = np.concatenate((P,ps,P))
        phase = np.concatenate((P,phis,P))
    else:
        power = ps
        phase = phis
    N = len(power)
    Nfft = np.int(2**np.ceil(np.log2(N)))
    Esase = np.sqrt(power)*np.exp(-1j*phase)
    t = np.array(range(len(Esase)))*deltaT
    energy0 = np.trapz(t,abs(Esase)**2)
    Ef = np.fft.fftshift(np.fft.fft(Esase,Nfft))
    f = Fs*(np.array(range(1,Nfft+1))*1.0-(Nfft+1)/2.0)/Nfft+f0
    lamda = c_speed/f
    energy1 = np.trapz(lamda,abs(Ef)**2)
    Nor = abs(energy0/energy1)
    Ef = np.sqrt(Nor)*Ef
    return Ef, f, lamda  
def readin(f2o):
    f = open(f2o)
    alllines=f.readlines()
    f.close()
    crystal = {}
    for line in alllines:
        if 'thickness' in line:
            crystal['thickness'] = float(line.split('=')[-1])
        if 'bragg' in line:
            crystal['bragg'] = float(line.split('=')[-1])
        if 'asymmetry' in line:
            crystal['asymmetry'] = float(line.split('=')[-1])
        if 'pho_energy' in line:
            crystal['pho_energy'] = float(line.split('=')[-1])
        if 'xr0' in line:
            crystal['xr0'] = float(line.split('=')[-1])
        if 'xi0' in line:
            crystal['xi0'] = float(line.split('=')[-1])
        if 'xrh' in line:
            crystal['xrh'] = float(line.split('=')[-1])
        if 'xih' in line:
            crystal['xih'] = float(line.split('=')[-1])
    crystal['ele_suscept0'] = crystal['xr0'] + 1j*crystal['xi0']
    crystal['ele_susceptH'] = crystal['xrh'] + 1j*crystal['xih']
    crystal['ele_susceptHBar'] = crystal['xrh'] - 1j*crystal['xih']
    return crystal

def transmission(crystal, f):
    scinum =sciconst()
    cry_thickness = crystal['thickness']        
    cry_asymmetry = crystal['asymmetry']     
    ele_suscept0  = crystal['ele_suscept0']         
    ele_susceptH  = crystal['ele_susceptH']  
    ele_susceptHbar  = crystal['ele_susceptHBar']
    h_Plank = scinum['h-plank'] 
    cry_bragg = crystal['bragg']
    pho_energy = crystal['pho_energy']
    e_charge = scinum['e-charge']
    c_speed = scinum['c-speed']
    freq_arry = f
    gamma0 = np.cos(np.deg2rad(cry_bragg+cry_asymmetry-90))          
    gammaH = np.cos(np.deg2rad(cry_bragg-cry_asymmetry+90))          
    asy_factor = gamma0/gammaH                  
    wavelength = h_Plank*c_speed/pho_energy/e_charge 
    ang_freq = 2*np.pi*c_speed/wavelength            
    wave_num = ang_freq/c_speed
    extin_len = np.sqrt(gamma0*np.abs(gammaH))/(wave_num*np.sqrt(ele_susceptH*ele_susceptHbar))
    A = cry_thickness/extin_len
    C = np.exp(1j*ele_suscept0*wave_num*cry_thickness/(2*gamma0))
    G = np.sqrt(np.abs(asy_factor)*(ele_susceptH*ele_susceptHbar))/ele_susceptHbar
    Omega = 2*np.pi*freq_arry-ang_freq
    tmp = -4*Omega*(np.sin(np.deg2rad(cry_bragg))**2/ang_freq)*(1-2*Omega/ang_freq)+ele_suscept0*(1-asy_factor)
    y = wave_num*extin_len/(2*gamma0)*(asy_factor*(tmp))
    Y1 = -y-np.sqrt(y**2+asy_factor/np.abs(asy_factor))
    Y2 = -y+np.sqrt(y**2+asy_factor/np.abs(asy_factor))
    R1 = G*Y1
    R2 = G*Y2
    R00 = np.exp(1j*(ele_suscept0*wave_num*cry_thickness/2/gamma0+A/2*Y1))*(R2-R1)/(R2-R1*np.exp(1j*A/2*(Y1-Y2)))
    R0H = R1*R2*(1-np.exp(1j*A/2*(Y1-Y2)))/(R2-R1*np.exp(1j*A/2*(Y1-Y2)))
    R001 = R00-C
    return R001, R00, R0H 
def timdomain(Esase, f, R00):
    scinum = sciconst()
    c_speed = scinum['c-speed']
    Ef= Esase*R00
    lamda = c_speed/f
    energy0 = np.abs(np.trapz(lamda,abs(Ef)**2))
    Etimedomain = np.fft.ifft(np.fft.ifftshift(Ef))
    amplitude = np.abs(Etimedomain)
    dtt = 1/(f[1]-f[0])
    tt = dtt/len(f)*np.arange(0,len(f))
    energy1 = np.abs(np.trapz(tt,amplitude**2))
    Ewake = Etimedomain*np.sqrt(energy0/energy1)
    return tt, Ewake
def cutseed(tt, Ewake,zsts,zdua):
    zend = zsts+zdua
    c_speed = 299792458
    z = c_speed*tt
    zstsid = np.abs(z-zsts).argmin()
    zendid = np.abs(z-zend).argmin()
    return z[zstsid-1:zendid+2]-z[zstsid-1],Ewake[zstsid-1:zendid+2]
def seedin(efield, zpos, lamda, sigx,wpath):
    header1 = '? VERSION=2.0\n'
    header2 = '? ZPOS        PRADO        ZRAYL        ZWAIST        PHASE\n'
    zray = np.pi*(sigx)**2/lamda
    p = np.abs(efield)**2
    phi = np.angle(efield)
    f = open(wpath+'seed.in','w')
    f.write(header1)
    f.write(header2)
    for i in range(len(zpos)):
        line = '%7E' %(zpos[i])+'        '+'%7E' %(p[i])+'        '+'%7E' %(zray)+'        '+'%7E' %(0)+'        '+'%7E' %(phi[i])+'\n'
        f.write(line)
    f.close
    
cfile14 = '/afs/slac.stanford.edu/u/ra/gzhou/susceptdb/slot14'
cfile16 = '/afs/slac.stanford.edu/u/ra/gzhou/susceptdb/slot16'
num = 16
seeds = ipseed(0,num)      
slots = [14,16]
fpath = os.getcwd()
zdua = 2.94113524320E-05 
for i in slots:
    for j in seeds:
        print(i,j)
        ipath = '/nfs/slac/g/beamphysics/gzhou/LCLS-II-HE/selfseedingclass/1300/10.0keV/slot'+str(i)+'/1st/caseid='+str(j)+'/'
        wpath = '/nfs/slac/g/beamphysics/gzhou/LCLS-II-HE/selfseedingclass/1300/10.0keV/slot'+str(i)+'/2nd/caseid='+str(j)+'/'
        os.chdir(ipath)
        ps, phis, pz, xlamds, nslice, zsep = ReadGenesis2('mod.out')
        Ef, f, lamda  = Spectrumf(ps,phis,xlamds,nslice, zsep)
        sigx = 22e-6
        if i == 14:
            cfile = cfile14
            zsts = 17e-6
            lamda = 1.24035e-10
        if i == 16:
            cilef = cfile16
            zsts = 17e-6
            lamda = 1.2403e-10
        crystal = readin(cfile)
        R001, R00, R0H = transmission(crystal, f)
        tt, Ewake = timdomain(Ef, f, R00)  
        zpos,efield = cutseed(tt, Ewake,zsts,zdua)
        seedin(efield, zpos, lamda, sigx,wpath)
        
        


    
    
