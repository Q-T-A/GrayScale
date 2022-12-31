'''
===============================================================================
ENGR 133 Fa 2020

Assignment Information
	Assignment:     Py1 Task 1
	Author:         Quinton austin, qaustin@purdue.edu
	Team ID:        LC#-04
	
Contributors:   Quinton Austin, qaustin@purdue.edu 
                Angel Avila, aavilago@purdue.edu
                Gage gardinier, Ggardini@purdue.edu 
                Christopher Ji, ji170@purdue.edu
	My contributor(s) helped me:	
	[ ] understand the assignment expectations without
		telling me how they will approach it.
	[ ] understand different ways to think about a solution
		without helping me plan my solution.
	[ ] think through the meaning of a specific error or
		bug present in my code without looking at my code.
	Note that if you helped somebody else with their code, you
	have to list that person as a contributor here as well.
===============================================================================
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

xk = np.array([-1,0,1,-2,0,2,-1,0,1])
yk = np.array([-1,-2,-1,0,0,0,1,2,1])
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def xkernal(im):
    #performs the dot product for the x direction
    total = 0
    for i in range(0,9):
        total += xk[i]*im[i]
    return total
        
def ykernal(im):
    #performs the dot product for the y direction
    total = 0
    for i in range(0,9):
        total += yk[i]*im[i]
    return total
def ref(im, x,y):
    #creates the 3x3 window that surrounds the desired pixel
     a =im[x-1,y-1]
     b = im[x,y-1]
     c = im[x+1,y-1]
     d = im[x-1,y]
     e =im[x,y]
     f = im[x+1,y]
     g = im[x-1,y+1]
     h = im[x,y+1]
     i = im[x+1,y+1]
     arr = np.array([a,b,c,d,e,f,g,h,i])
     return arr
 
def gradientcalc(newarrx, newarry):
    #calculates gradient
    gradient = newarrx.copy()
    for y in range(0,newarrx.shape[1]-1):
        for x in range(0,newarrx.shape[0]-1):
            gradient[x][y] = (newarrx[x][y]**2 + newarry[x][y]**2) **.5
    return gradient
         
         
def parse(im):
    #copy array so the original is unchanged
    newarrx = np.copy(im)
    newarry = np.copy(im)
    #goes through the array and accesses each index
    for y in range (0,im.shape[1]-1):
        for x in range (0,im.shape[0]-1):
            #runs sobel edge enhancement
            #replaces current index without changing 
            #the reference image's data
            newarrx[x][y] = xkernal(ref(im,x,y))
            newarry[x][y] = ykernal(ref(im,x,y))
    return newarrx,newarry
img = mpimg.imread('itachi_minimal.png')
rawgray = rgb2gray(img)    
twowayedge = parse(rawgray)
gray = gradientcalc(twowayedge[0],twowayedge[1])
plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
plt.show()


#def sobel_edge_detection(im): 
'''  
thresh1 = .5
#thresh2 = .9

#Load image
im = mpimg.imread('Coins.png')
#im = rgb2gray(gim)

gryim = np.mean(im[:,:,0:2],2)
region1 =  (thresh1<gryim)
#region2 =  (thresh2<gryim)
nregion1 = ~ region1
#nregion2 = ~ region2

#Plot figure and two regions
fig, axs = plt.subplots(2,2)
axs[0,0].imshow(im)
axs[0,1].imshow(region1)
#axs[1,0].imshow(region2)


#Get location of edge by comparing array to it's 
#inverse shifted by a few pixels
shift = -2
edgex1 = (region1 ^ np.roll(nregion1,shift=shift,axis=0))
edgey1 = (region1 ^ np.roll(nregion1,shift=shift,axis=1))
#edgex2 = (region2 ^ np.roll(nregion2,shift=shift,axis=0)) 
#edgey2 = (region2 ^ np.roll(nregion2,shift=shift,axis=1))

#Plot location of edge over image
axs[1,1].imshow(im)
axs[1,1].contour(edgex1,2,colors='w',lw=2.)
axs[1,1].contour(edgey1,2,colors='w',lw=2.)
#axs[1,1].contour(edgex2,2,colors='g',lw=2.)
#axs[1,1].contour(edgey2,2,colors='g',lw=2.)
plt.show()
                    # ,cmap = 'gray', interpolation = 'none')
                    '''
'''
===============================================================================
ACADEMIC INTEGRITY STATEMENT
    I have not used source code obtained from any other unauthorized
    source, either modified or unmodified. Neither have I provided
    access to my code to another. The project I am submitting
    is my own original work.
===============================================================================
'''