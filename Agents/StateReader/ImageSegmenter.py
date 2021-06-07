#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 08:48:24 2019

The ImageSegmenter reimplementation in Python

@author: chengxue
"""
import numpy as np
import cv2
from scipy import signal
from StateReader.cv_utils import Rectangle
import os
import time
import matplotlib.pylab as plt
from scipy.ndimage import gaussian_filter
from skimage import measure

class ImageSegmenter:
    
    def __init__(self,screenshot):
        
        self._groundLevel = 0
        
        self.wh = 220
        self.ws = 0.5
        self.wv = 1
        
        self.EDGE_BOUND = 180
        
        #objects in game
        self.screenshot = screenshot
        self.BACKGROUND = 0 
        self.GROUND = 1 
        self.HILLS = 2 
        self.SLING = 3 
        self.RED_BIRD = 4 
        self.YELLOW_BIRD = 5 
        self.BLUE_BIRD = 6 
        self.BLACK_BIRD = 7 
        self.WHITE_BIRD = 8 
        self.PIG = 9 
        self.ICE = 10 
        self.WOOD = 11 
        self.STONE = 12 
        self.DUCK = 13 
        self.EDGE = 14 
        self.WATERMELON = 15
        self.TRAJECTORY = 16
        self.CLASS_COUNT = 17
        
        self.MAX_DIST = 100
        
        self.EDGE_THRESHOLD1 = 250
        
        
        self.EDGE_THRESHOLD2 = 100
        
        #training color points
        self._trainData = [
                
        [72, 92, 45, self.GROUND],
        [38, 55, 32, self.GROUND],
        [29, 41, 26, self.GROUND],
        [115, 145, 78, self.GROUND],
        [121, 150, 82, self.GROUND],
        [200, 225, 151, self.GROUND],
        
        [64, 40, 24, self.HILLS],
        [48, 32, 16, self.HILLS],
        [144, 112, 80, self.HILLS],
        [104, 72, 48, self.HILLS],
        [120, 88, 64, self.HILLS],
        [160, 120, 88, self.HILLS],
        [88, 56, 40, self.HILLS],
        [72, 48, 32, self.HILLS],
        [48, 16, 8, self.SLING],
        [80, 40, 8, self.SLING],
        [120, 64, 32, self.SLING],
        [160, 112, 48, self.SLING],
        [200, 144, 88, self.SLING],
        [152, 88, 40, self.SLING],
        [176, 128, 64, self.SLING],
        [168, 112, 64, self.SLING],
        [208, 0, 40, self.RED_BIRD],
        [240, 216, 32, self.YELLOW_BIRD],
        [232, 176, 0, self.YELLOW_BIRD],
        [96, 168, 192, self.BLUE_BIRD],
        [80, 144, 168, self.BLUE_BIRD],
        [232, 232, 200, self.WHITE_BIRD],
        [248, 184, 72, self.WHITE_BIRD],
        [104, 224, 72, self.PIG],
        [160, 232, 0, self.PIG],
        [88, 176, 32, self.PIG],
        [110, 176, 12, self.PIG],
        [56, 104, 8, self.PIG],
        [88, 192, 240, self.ICE],
        [104, 192, 240, self.ICE],
        [120, 200, 240, self.ICE],
        [192, 240, 248, self.ICE],
        [144, 216, 240, self.ICE],
        [136, 208, 248, self.ICE],
        [128, 208, 248, self.ICE],
        [168, 224, 248, self.ICE],
        [69, 163, 187, self.ICE],
        [40, 160, 224, self.ICE],
        [176, 224, 248, self.ICE],
        [224, 144, 32, self.WOOD],
        [248, 184, 96, self.WOOD],
        [192, 112, 32, self.WOOD],
        [176, 88, 32, self.WOOD],
        [168, 96, 8, self.WOOD],
        [232, 160, 72, self.WOOD],
        [248, 192, 96, self.WOOD],
        [248, 176, 72, self.WOOD],
        [156, 112, 83, self.WOOD],
        [168, 88, 32, self.WOOD],
        [255, 255, 255, self.BACKGROUND],
        [248, 224, 80, self.DUCK],
        [248, 208, 32, self.DUCK],
        [248, 136, 32, self.DUCK],
        [240, 176, 16, self.DUCK],
        [128, 168, 24, self.WATERMELON],
        [88, 112, 16, self.WATERMELON],
        [56, 88, 16, self.WATERMELON]
    ]
        
        #############first time needed#############
        #initialise drawing colors
        self._drawColor = {}
        self._colors = np.zeros(self.CLASS_COUNT)
        self._drawColor[self.BACKGROUND] = 0xdddddd
        self._drawColor[self.GROUND] = 0x152053 
        self._drawColor[self.HILLS] = 0x342213
        self._drawColor[self.SLING] = 0x7f4120
        self._drawColor[self.EDGE] = 0x000000
        self._drawColor[self.STONE] = 0xa0a0a0
        self._drawColor[self.ICE] = 0x6ecdf8
        self._drawColor[self.WOOD] = 0xe09020
        self._drawColor[self.PIG] = 0x60e048
        self._drawColor[self.TRAJECTORY] = 0xffffff
        self._drawColor[self.BLUE_BIRD] = 0x60a8c0
        self._drawColor[self.RED_BIRD] = 0xd00028
        self._drawColor[self.YELLOW_BIRD] = 0xf0d820 
        self._drawColor[self.BLACK_BIRD] = 0x0f0f0f
        self._drawColor[self.WHITE_BIRD] = 0xe8e8c8
        self._drawColor[self.DUCK] = 0xf0d820
        self._drawColor[self.WATERMELON] = 0x80a818
        
        #initialse minimum and maximum sizes
        self.MIN_SIZE = np.zeros((self.CLASS_COUNT))
        self.MAX_SIZE = np.zeros((self.CLASS_COUNT))
        
        for i in range(self.CLASS_COUNT):
            self.MIN_SIZE[i] = 15
            self.MAX_SIZE[i] = 4000
        

        self.MIN_SIZE[self.PIG] = 65
        self.MIN_SIZE[self.HILLS] = 250
        self.MIN_SIZE[self.SLING] = 200
        self.MIN_SIZE[self.BLUE_BIRD] = 10
        self.MIN_SIZE[self.RED_BIRD] = 20
        self.MIN_SIZE[self.YELLOW_BIRD] = 20
        self.MIN_SIZE[self.BLACK_BIRD] = 20
        self.MIN_SIZE[self.TRAJECTORY] = 1
        
        self.MAX_SIZE[self.HILLS] = 1000000
        self.MAX_SIZE[self.BLUE_BIRD] = 30
        self.MAX_SIZE[self.TRAJECTORY] = 60
        
        t = time.time()
        #assign all of the 15bits colors to a particular class
        self._assignedType = np.zeros((1<<15))
        
        for color in range(1<<15):
            self._assignedType[color] = self._assignType(color)
        #print('assign type took: {0:.4f}'.format(time.time()-t))
        #############first time needed#############
        
        
        #parse the screenshot
        self._width = screenshot.shape[1]
        self._height = screenshot.shape[0]
        self._image = self._compressImage()
        
        
        #process hsv
        self.hsv = np.zeros((self._height,self._width,3))
        self.hsv = cv2.cvtColor(self.screenshot,cv2.COLOR_BGR2HSV).astype(int)
        
        
        
        #classify the pixcels and assign a label to each pixcel
        t = time.time()
        self._class = self._classifyPixcels()
#        print('classification took: {0:.4f}'.format(time.time()-t))
        
        t = time.time()
        self._groundLevel = self.findGroundLevel()
#        print('find ground level took: {0:.4f}'.format(time.time()-t))
        
    def findConnectedComponents(self):
        '''
        find connected components in the class map
        '''
        self._findEdges()
        ind = (self.isEdge == 1) & ((self._class == self.ICE) | (self._class == self.WOOD) | (self._class == self.STONE))
        self._class[ind] = self.EDGE
        self._class = self._class.astype(np.uint8)
        
        
        #then find the connected components for _class
        ret = measure.label(self._class, connectivity=2)
        Nseg = np.max(ret) + 1
        for i in range(Nseg):

            ret[ret == i] = self._class[ret == i][0]

            
        ret = ret.astype(np.uint8)
        self.ConnectedComponents = ret
        return ret
        
    def _findEdges(self):

           
        #for horizontal edge, we have wh*(h_{x_0-1,y_0} - h_{x0,y_0})**2 + ws*(s_{x_0+1,y_0} - s_{x_0,y_0})**2
        undefined_h = (self.screenshot[:,:,0] == self.screenshot[:,:,1]) & (self.screenshot[:,:,1] == self.screenshot[:,:,2])
        #self.hsv[:,:,1][undefined_h] = 0 # change sat to 0 

        
        strenth = np.zeros((self._height,self._width,4))
        ind_range = []
        t = time.time()
        #horizental range
        h_diff_f = self.hsv[1:-1,:,0] - self.hsv[:-2,:,0] 
        h_diff_b = self.hsv[1:-1,:,0] - self.hsv[2:,:,0]
        s_diff_f = self.hsv[1:-1,:,1] - self.hsv[:-2,:,1] 
        s_diff_b = self.hsv[1:-1,:,1] - self.hsv[2:,:,1]
        
        
        
        ind_range.append([h_diff_f,h_diff_b,s_diff_f,s_diff_b])
        
        #45 range
        h_diff_f = self.hsv[1:-1,1:-1,0] - self.hsv[:-2,2:,0] 
        h_diff_b = self.hsv[1:-1,1:-1,0] - self.hsv[2:,:-2,0]
        s_diff_f = self.hsv[1:-1,1:-1,1] - self.hsv[:-2,2:,1] 
        s_diff_b = self.hsv[1:-1,1:-1,1] - self.hsv[2:,:-2,1]
        ind_range.append([h_diff_f,h_diff_b,s_diff_f,s_diff_b])
        
        #vertical range
        h_diff_f = self.hsv[:,1:-1,0] - self.hsv[:,:-2,0] 
        h_diff_b = self.hsv[:,1:-1,0] - self.hsv[:,2:,0]
        s_diff_f = self.hsv[:,1:-1,1] - self.hsv[:,:-2,1] 
        s_diff_b = self.hsv[:,1:-1,1] - self.hsv[:,2:,1]
        ind_range.append([h_diff_f,h_diff_b,s_diff_f,s_diff_b])
        
        #135 range
        h_diff_f = self.hsv[1:-1,1:-1,0] - self.hsv[:-2,:-2,0] 
        h_diff_b = self.hsv[1:-1,1:-1,0] - self.hsv[2:,2:,0]
        s_diff_f = self.hsv[1:-1,1:-1,1] - self.hsv[:-2,:-2,1] 
        s_diff_b = self.hsv[1:-1,1:-1,1] - self.hsv[2:,2:,1]
        ind_range.append([h_diff_f,h_diff_b,s_diff_f,s_diff_b])

        order = ['hori','45','vert','135']
        for i in range(len(ind_range)):
            strenth[:,:,i] = self._distance_parallel(*ind_range[i],order[i])
        
#        for underfined h points
        
        undfined_list = []
        value_diff_f = self.hsv[1:-1,:,2] - self.hsv[:-2,:,2] 
        value_diff_b = self.hsv[1:-1,:,2] - self.hsv[2:,:,2]
        undfined_list.append([value_diff_f,value_diff_b])
        
        value_diff_f = self.hsv[1:-1,1:-1,2] - self.hsv[:-2,2:,2] 
        value_diff_b = self.hsv[1:-1,1:-1,2] - self.hsv[2:,:-2,2]
        undfined_list.append([value_diff_f,value_diff_b])
        
        value_diff_f = self.hsv[:,1:-1,2] - self.hsv[:,:-2,2] 
        value_diff_b = self.hsv[:,1:-1,2] - self.hsv[:,2:,2]
        undfined_list.append([value_diff_f,value_diff_b])
        
        value_diff_f = self.hsv[1:-1,1:-1,2] - self.hsv[:-2,:-2,2] 
        value_diff_b = self.hsv[1:-1,1:-1,2] - self.hsv[2:,2:,2]
        undfined_list.append([value_diff_f,value_diff_b])
        
        for i in range(len(ind_range)):
            distance1 = self.wv * np.abs(undfined_list[i][0]) 
            distance2 = self.wv * np.abs(undfined_list[i][1])
            
            distance1[distance1>self.EDGE_BOUND] = self.EDGE_BOUND
            distance2[distance2>self.EDGE_BOUND] = self.EDGE_BOUND
            
            distance = distance1 + distance2
            
            if i == 0:
                distance = np.pad(distance,((1,1),(0,0)),'constant',constant_values=0)
                distance = distance * 2
                
            elif i == 2:
                distance = np.pad(distance,((0,0),(1,1)),'constant',constant_values=0)
                distance = distance * 2
            
            else:
                distance = np.pad(distance,((1,1),(1,1)),'constant',constant_values=0)
            

            strenth[:,:,i][undefined_h] = distance[undefined_h]
        
        #return strenth

        #cross-correlate with neigbouring points
        
        #average the neigbours in perpendicular direction
        #this is in fact average the nearby points
        #which is nothing but moving average across the 4 directions
        
        cross_correlate = np.zeros((self._height,self._width,4))
        
        for o in range(4):
            
                if o == 0:
                    cov = np.array(
                            #[[0,0,0],
                             [0.333,0.333,0.333]).reshape(1,3)#,
                             #[0,0,0]]).reshape(3,3)
                    cross_correlate[:,:,o] = signal.correlate2d(strenth[:,:,o],cov,mode='same',boundary='fill',fillvalue=0)     
                              

                if o == 2:
                    cov = np.array(
                            #[[0,0,0],
                             [0.333,0.333,0.333]).reshape(3,1)#,
                             #[0,0,0]]).reshape(3,3)
            
                    cross_correlate[:,:,o] = signal.correlate2d(strenth[:,:,o],cov,mode='same',boundary='fill',fillvalue=0)           
                
                if o == 1:
                    cov = np.array(
                            [[0.333,0,0],
                             [0,0.333,0],
                             [0,0,0.333]]).reshape(3,3)   
                    
                    cross_correlate[:,:,o] = signal.correlate2d(strenth[:,:,o],cov,mode='same',boundary='fill',fillvalue=0) 

                if o == 3:
                    cov = np.array(
                            [[0,0,0.333],
                             [0,0.333,0],
                             [0.333,0,0]]).reshape(3,3)                   
                    
                    cross_correlate[:,:,o] = signal.correlate2d(strenth[:,:,o],cov,mode='same',boundary='fill',fillvalue=0)

        #apply non-maximum suppression for each direction
        
        non_max = np.zeros((self._height,self._width,4))
        
        #horizental
        for i in range(1,cross_correlate.shape[0]-1):
            non_max[i,:,0] = cross_correlate[i,:,0] * ((cross_correlate[i,:,0] >= cross_correlate[i+1,:,0]) & (cross_correlate[i,:,0] > cross_correlate[i-1,:,0]))

#        non_max[:,:,0] = cross_correlate[:,:,0]
        #45
        for i in range(1,cross_correlate.shape[0]-1):
            non_max[i,1:-1,3] = cross_correlate[i,1:-1,3] * ((cross_correlate[i,1:-1,3] >= cross_correlate[i+1,:-2,3]) & (cross_correlate[i,1:-1,3] > cross_correlate[i-1,2:,3]))
        
        #vertical
        for i in range(1,cross_correlate.shape[1]-1):
            non_max[:,i,2] = cross_correlate[:,i,2] * ((cross_correlate[:,i,2] >= cross_correlate[:,i+1,2]) & (cross_correlate[:,i,2] > cross_correlate[:,i-1,2]))
            
        #135
        for i in range(1,cross_correlate.shape[0]-1):
            non_max[i,1:-1,1] = cross_correlate[i,1:-1,1] * ((cross_correlate[i,1:-1,1] >= cross_correlate[i+1,2:,1]) & (cross_correlate[i,1:-1,1] > cross_correlate[i-1,:-2,1]))
           

        non_max[non_max<self.EDGE_THRESHOLD2] = 0
                
        isEdge = np.zeros((self._height,self._width,4))
        
        threshold1_points = np.where(non_max>=self.EDGE_THRESHOLD1)
        
        
        isEdge[threshold1_points] = True
        
        for point in range(len(threshold1_points[0])):
            
            check_set = {(threshold1_points[0][point],
                          threshold1_points[1][point],
                          threshold1_points[2][point])}
    
            while len(check_set) != 0:
                
                point_to_check = check_set.pop()
                
                for y in [-1,1]:
                    for x in [-1,1]:
                        if y == 0 and x == 0:
                            continue
                        elif point_to_check[0] + y >= 0 and point_to_check[0] + y < 480 and \
                        point_to_check[1] + x >= 0 and point_to_check[1] + x < 840:
                            point_around = point_to_check[0] + y, point_to_check[1] + x , point_to_check[2]
                            
                            if non_max[point_around] > self.EDGE_THRESHOLD2 and isEdge[point_around] != True:
                                isEdge[point_around] = True
                                check_set.add(point_around)
                    
        # combine edge in all four directions
        isEdge = isEdge[:,:,0]+isEdge[:,:,1]+isEdge[:,:,2]+isEdge[:,:,3]
        isEdge[isEdge>0] = 1
        isEdge[:2,:] = 0
        isEdge[:,-2:] = 0
        isEdge[:,:2] = 0
        isEdge[:,-2:] = 0
        self.isEdge = isEdge
#        print(time.time()-t)
        return isEdge
    
    def _distance_parallel(self,h_diff_f,h_diff_b,s_diff_f,s_diff_b,mode):
        '''
        mode can be hori vert 45 and 135
        '''
        distance_1 = np.sqrt(self.wh * h_diff_f ** 2 + self.ws * s_diff_f ** 2)
        distance_1[distance_1>self.EDGE_BOUND] = self.EDGE_BOUND
        distance_2 = np.sqrt(self.wh * h_diff_b ** 2  + self.ws * s_diff_b ** 2)
        distance_2[distance_2>self.EDGE_BOUND] = self.EDGE_BOUND
        
        

        strength = distance_1 + distance_2 
        
        if mode == 'hori':
            strength = np.pad(strength,((1,1),(0,0)),'constant',constant_values=0)
            strength = strength * 1.5
            
        elif mode == 'vert':
            strength = np.pad(strength,((0,0),(1,1)),'constant',constant_values=0)
            strength = strength * 1.5
        
        else:
            strength = np.pad(strength,((1,1),(1,1)),'constant',constant_values=0)
                      
        return strength
        
        
    def findGroundLevel(self):
        '''
        return value in y direction
        
        ?might need to be reimplemented
        
        '''
        if self._groundLevel != 0:
            return self._groundLevel
        
        mask = np.where(self._class == self.GROUND)
        
        
        zero_martix = np.zeros((self._height,self._width))
        
        zero_martix[mask] = 1
        
        #sum the number of ground horizentally
        
        ground_count = np.sum(zero_martix,1)
        
        #some predefine condition
        mask = self._width * 0.8
        
        #ground_count[ground_count>=mask] = 0
        
        #print(ground_count[ground_count>0])
        
        #return the largest level
        
        return np.argmax(ground_count)
        
        
    def _classifyPixcels(self):
        '''
        classify the compressed image pixcel values to class
        '''
        
        #self._class = np.zeros((self._height,self._width))
        
        return  np.array(list(map(lambda x : self._assignedType[x],self._image)))
        

     
    def _assignType(self,color):
        '''
        assign color to the nearest class
        '''
        
        ptype = self.BACKGROUND
        minDist = 999999
        
        # get r g b values
        r = (color >> 10) << 3
        g = ((color >> 5) & 31 ) << 3
        b = (color & 31) << 3
        
        #special cases where pixel is greay scale
        
        if r == g and r == b:
            if r >= 88 and r <= 208:
                return self.STONE
            elif r > 232:
                return self.TRAJECTORY
            elif r == 64:
                return self.BLACK_BIRD
        
        for i in range(len(self._trainData)):
        
            d1 = r - self._trainData[i][0]
            d2 = g - self._trainData[i][1]
            d3 = b - self._trainData[i][2]
            
            dist =  d1**2 + d2**2 + d3**2
            
            if dist < minDist and dist < self.MAX_DIST**2:
                minDist = dist
                ptype = self._trainData[i][3]
        
        return ptype
        
    def _compressImage(self):
        '''
        compress screenshot by taking the first 5bits of each r g b color
        and combine them to a new 15 bits image
        '''
        
        mask = np.ones((self._height,self._width,3)).astype(int) * 0b11111000
        mask = self.screenshot & mask
        mask[:,:,0] = mask[:,:,0] << 7
        mask[:,:,1] = mask[:,:,1] << 2
        mask[:,:,2] = mask[:,:,2] >> 3
        return mask[:,:,0] | mask[:,:,1] | mask[:,:,2]        


    
            
if __name__ == "__main__":
    def int2img(array):
        h = array.shape[0]
        w = array.shape[1]
        ret = np.zeros((h,w,3))
        max_value = np.max(array)
        min_value = np.min(array)
        array = (array-min_value) * (255.0/(max_value-min_value))
        #print(max_value)
        ret[:,:,0] = array
        ret[:,:,1] = array
        ret[:,:,2] = array
        return ret.astype(int)    
    #path = os.listdir('./test_screenshot/')
    while True:
        img = cv2.imread('../demo/'+'screenshot.png')
        t = time.time()
        img = img[:,:,::-1]
        imageSeg = ImageSegmenter(img)
        ret = imageSeg.findConnectedComponents()
        print('total time required: {0:.2f}'.format(time.time()-t))
        img = img[:,:,::-1]
        for c in range(2,13):
            if c!=7:
                to_ret = np.zeros((imageSeg._height,imageSeg._width)).astype(np.uint8)
                to_ret[ret==c] = 255
                contours, hierarchy = cv2.findContours(to_ret, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                for con in contours:
                    x,y,w,h = cv2.boundingRect(con)
    
                    if cv2.contourArea(con) >= imageSeg.MIN_SIZE[c] and cv2.contourArea(con) <= imageSeg.MAX_SIZE[c]:
                        if x >= 50 and y >=100:
                            print(x,y,c)
                            print(cv2.contourArea(con))
                            if c in [4,5,6,7,8,9]:
                                (x,y),radius = cv2.minEnclosingCircle(con)
                                center = (int(x),int(y))
                                radius = int(radius) + 2
                                cv2.circle(img,center,radius,(0,0,255),1)
                            elif c in [3]:
                                
                                if c == 3 and x < 240 and y > 240:
                                    box = Rectangle((np.array([y+h,y]),np.array([x+w,x])))
                                    
                                    box.width,box.height = box.height,box.width
                                    
                                    #check the aspect ration
                                    if box.height/ box.width>2:
                                
                                        rect = cv2.minAreaRect(con)
                                        box = cv2.boxPoints(rect)
                                        box = np.int0(box)
                                        cv2.drawContours(img,[box],0,(0,0,255),1)
                            elif c in [2]:
                                cv2.drawContours(img, con, -1, (0, 0, 255), 1)
                            else:
                                hull = cv2.convexHull(con)
                                cv2.drawContours(img,[hull],0,(0,0,255),1)
                        
        cv2.imshow('res',img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()