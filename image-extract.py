'''
Reads experimental images produced from *_centroid_circle_plot*.vi
The raw data is presented in the form:
#spots channel
x y radius area ! spot 1
= = ====== ==== ! spot 2
= = ====== ==== ! spot n, n = #spots
The area of the pixel is not used in any calculation!
Written by Josh Featherstone 2021
'''

data = r'/home/josh/Downloads/Brouard/Images/argon 7'
#The plinth constant (pc) determines how the pixels are counted
#Using a pc of 0 counts each pixel as a single intensity ignoring radius
#Using a pc of 1 determines pixels by the radius specified in the file
#Using a pc >= 2 will set the splat radius to that specific size
pc = 2
#What is the camera rotation? Only change this if the value has changed
cr = 102

'''!!! Don't edit past this point unless you know what you're doing !!!'''

import math
import os
import time
import sys

import matplotlib.pyplot as plt
import numpy as np
#When we calculate normalized difference we will divide by 0, so ignore error
np.seterr(divide='ignore', invalid='ignore')

from datetime import datetime
from scipy.ndimage import rotate

start = time.time()

#To start off calculate splats for image intensity
def splat_count(splat_width):
    '''Math from fortran script by Sean D.S Gordon 2013-16'''
    #Calculate the splat radius
    sr  = int(math.sqrt(-math.log(0.05)/math.log(2)) * splat_width)
    if(sr > 50): sr = 50
    splat_const  = (1.0/splat_width/splat_width)*math.log(2)
    splat = np.zeros(((sr*2)+1,(sr*2)+1))
    for NX in range(-sr,sr+1):
        for NY in range(-sr,sr+1):
            splat[NX+sr][NY+sr] = math.exp(-((NX*NX)+(NY*NY))*splat_const)
    return sr,splat

#If calculating using 1 intensity no need for splat
if pc == 0: splat = []
#If intensity by radius in file calculate splats for radius of 1 through 100
#Faster to precalculate then to calculate at each loop
elif pc == 1: 
    splat_consts = [splat_count(x) for x in range(1,101)]
    splat_consts.insert(0,[0.,0.]) #Add 0th index for radius indexing
else: splat = splat_count(pc*0.5) #if hard set only one index to use

#Where channel 1 is channels[0], etc.
channels = np.zeros((8,1500,1500),dtype=float)
#How many pixels per channel?
chan_hits = [0 for _ in range(9)]
#Set some empty parameters
c = 0 #default channel to 0
max_x, max_y = 0, 0 #set default maximums
scans = 0

with open(data,'r') as opf:
    for line in opf:
        line = line.split()
        #Lines of length two are the number of pixels + channel name
        if len(line) == 2:
            #if scans == 1: break
            c = int(line[1].strip())-1 #remove \n and -1 for indexing
            scans+=1
        else:
            #Change all values to int and subtract 1
            line = [int(float(x.strip()))-1 for x in line]
            x,y,r = line[0],line[1],line[2]
            #Check if x and y are larger
            if x > max_x: max_x = line[0]
            if y > max_y: max_y = line[1]
            chan_hits[c]+=1
            #If pc is set to 0 add 1 to each pixel found in scan
            if pc == 0:
                channels[c][x][y] += 1
                continue
            #If using file radius values grab radius data
            if pc == 1: splat = splat_consts[r]
            sr = splat[0]
            sa = np.array(splat[1])
            #lowest x and y followed by largest x and y indexes
            xl, yl, xm, ym = int(x-sr), int(y-sr), int(x+sr+1), int(y+sr+1)
            while xl < 0: #If intensity around x = 0 
                sa = np.delete(sa,0,0)
                xl+=1
            while yl < 0: 
                sa = np.delete(sa,0,1)
                yl+=1
            #Add intensity to channel at position X and Y
            channels[c][xl:xm,yl:ym] += sa

#Set the larger value to the smaller value
#Do this so that we have square images
if max_x > max_y: max_x = max_y
if max_y > max_x: max_y = max_x

#Remove placeholders columns and rows from data
tchannels = np.zeros((8,max_x,max_y),dtype=float)
for c in range(len(channels)):
    ta = channels[c]
    ta = np.delete(channels[c], slice(max_y, 1500), 1)
    ta = np.delete(ta, slice(max_x, 1500), 0)
    tchannels[c] = ta
channels = tchannels

#Make a str variable to store all our output data
now = datetime.now()
string = f'File was produced on : {now.strftime("%d/%m/%Y %H:%M:%S")}\n'
string += f'The plinth constant used was : {pc}\n'
string += f'The camera rotation used was : {cr}\n\n'
string += 'STATISTICAL INFORMATION\n'
for i in range(len(channels)):
    string+=f' Channel {i+1} : {chan_hits[i]}\n'
string+='--------------------\n'
string+=f' Total hit : {sum(chan_hits)}\n\n'

#Determine symmetry
sig_NO=(chan_hits[2]+chan_hits[3])-(chan_hits[1]+chan_hits[0])
sig_ON=(chan_hits[6]+chan_hits[7])-(chan_hits[4]+chan_hits[5])

try: string+=f' Asymmetry : {round(100.0*(sig_NO-sig_ON)/(sig_NO+sig_ON),2)}\n'
except ZeroDivisionError: string+='\n Asymmetry : 0.0\n'
sig_ON=1.04*sig_ON #steric asymmetry based on experimental unscattered signal
try: string+=f'#Asymmetry : {round(100.0*(sig_NO-sig_ON)/(sig_NO+sig_ON),2)}\n'
except ZeroDivisionError: string+='#Asymmetry : 0.0\n'

#Rebuild image data
Himage_NO = (channels[0] - channels[2]) #H_NO
Vimage_NO = (channels[3] - channels[1]) #V_NO

Himage_ON = (channels[4] - channels[6]) #H_ON
Vimage_ON = (channels[7] - channels[5]) #V_ON

#Get sums of all of the arrays
HNOS, VNOS = int(np.sum(Himage_NO)), int(np.sum(Vimage_NO))
HONS, VONS = int(np.sum(Himage_ON)), int(np.sum(Vimage_ON))

string+='\nIMAGE INTENSITY INFORMATION\n'

string+=f' Num scans : {scans}\n'
string+=f' Himage_NO : {HNOS}\n'
string+=f' Vimage_NO : {VNOS}\n'
string+=f' Himage_ON : {HONS}\n'
string+=f' Vimage_ON : {VONS}\n'
string+='--------------------\n'
string+=f'Brightness : {HNOS+VNOS+HONS+VONS}\n\n'

sig_NO=HNOS+VNOS
sig_ON=HONS+VONS

try: string+=f' Asymmetry : {round(100.0*(sig_NO-sig_ON)/(sig_NO+sig_ON),2)}\n'
except ZeroDivisionError: string+='\n Asymmetry : 0.0\n'
sig_ON=1.04*sig_ON #steric asymmetry based on experimental unscattered signal
try: string+=f'#Asymmetry : {round(100.0*(sig_NO-sig_ON)/(sig_NO+sig_ON),2)}\n'
except ZeroDivisionError: string+='#Asymmetry : 0.0\n'

string+='\nWriting image data to files...\n\n'

def largest_intensity(intensities):
    '''Calculate average percent of largest intensity compared to 
    nearby intensities.'''
    percents = intensities/intensities[0]
    if round(np.average(percents),2) >= 0.8: return True
    return False

def save_image(img_data,fname):
    '''Used to save the various images obtained.'''
    X,Y = [x for x in range(max_x)],[x for x in range(max_y)]
    X,Y = np.meshgrid(Y,X)
    Z = img_data.reshape(max_x,max_y)
    Z = rotate(Z, cr, reshape=False) #Rotate to camera rotation

    #Get maximums by sorting each array
    ZM = np.sort(Z,axis=1)
    #Take largest value from each array
    ZM = sorted(ZM[:,-1],reverse=True)
    tmf = False #true maximum found?
    while tmf == False:
        tmf = largest_intensity(ZM[:10])
        if tmf == False: del ZM[0]
    Z[Z<0] = 0 #Set everything below 0 to 0
    Z[Z>ZM[0]] = ZM[0] #Scale everything to best intensity

    fig, ax = plt.subplots(nrows=1,ncols=1)
    im = ax.pcolormesh(X,Y,Z,shading='auto')
    fig.colorbar(im)
    #shrink image to account for rotation
    plt.ylim([max_x*0.07, max_x-max_x*0.07])
    plt.xlim([max_x*0.07, max_x-max_x*0.07])
    plt.savefig(fname)

#Create directory to store data
dname = os.path.dirname(data) #directory name
fname = os.path.basename(data) #filename
fname = fname.replace(' ','_')+'_d' #Add _d in the event user used _
dname = os.path.join(dname,fname) #new directory name
try: os.mkdir(dname)
except FileExistsError: pass

#Save the various V & H images
if np.sum(Himage_NO) > 0 : save_image(Himage_NO,os.path.join(dname,'H_NO.png'))
if np.sum(Vimage_NO) > 0 : save_image(Vimage_NO,os.path.join(dname,'V_NO.png'))
if np.sum(Himage_ON) > 0 : save_image(Himage_ON,os.path.join(dname,'H_ON.png'))
if np.sum(Vimage_ON) > 0 : save_image(Vimage_ON,os.path.join(dname,'V_ON.png'))

#Build additional arrays for experimental image analysis
#Don't take the sum and difference of images if empty arrays
if np.sum(Vimage_NO) != 0 and np.sum(Himage_NO) != 0:
    sumimg_NO = Vimage_NO + Himage_NO #V+H_NO
    difimg_NO = Himage_NO - Vimage_NO #V-H_NO
    ndif_NO = difimg_NO/sumimg_NO #V-H_V+H_NO
    ndif_NO = np.nan_to_num(ndif_NO) #Replace nan values with 0

    save_image(sumimg_NO,os.path.join(dname,'V+H_NO.png'))
    save_image(difimg_NO,os.path.join(dname,'V-H_NO.png'))
    save_image(ndif_NO,os.path.join(dname,'V-H_V+H_NO.png'))

if np.sum(Vimage_ON) != 0 and np.sum(Himage_ON) != 0:
    sumimg_ON = Vimage_ON + Himage_ON #V+H_ON
    difimg_ON = Himage_ON - Vimage_ON #V-H_ON
    ndif_ON = difimg_ON/sumimg_ON #V-H_V+H_ON
    ndif_ON = np.nan_to_num(ndif_ON) #Replace nan values with 0

    save_image(sumimg_ON,os.path.join(dname,'V+H_ON.png'))
    save_image(difimg_ON,os.path.join(dname,'V-H_ON.png'))
    save_image(ndif_ON,os.path.join(dname,'V-H_V+H_ON.png'))

string+=f'Runtime (s): {round(time.time()-start,2)}'

tfn = os.path.join(dname,f'{fname[:-2]}.txt') #output name
with open(tfn,'w') as opf:
    opf.write(string)

print(string)
