#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 08:23:46 2019

@author: chengxue
"""

import os
#import matplotlib.pylab as plt
import time

import cv2
#from PIL import ImageGrab
import numpy as np
from skimage import measure

from StateReader.cv_utils import Rectangle
from StateReader.game_object import GameObject, GameObjectType

class VisionMBR:

    def __init__(self,screenshot):
        #print('asdfasdfads')
        self.screenshot = screenshot
        self._nHeight = screenshot.shape[0] #height of the scene
        self._nWidth = screenshot.shape[1] #width of the scene
        self._scene = np.zeros((self._nHeight,self._nWidth)) #quantized scene colours
        self._nSegments = 0 #number of segments
        self._colours = np.zeros(self._nSegments) # colour for each segment
        self._boxes = [] # bounding box list for each segment
        self._regionTreshold = 20  # minimal pixels in a region

        #this stores the value of the mail color and associate color
        #for all objects
        self._color_values = {
                            'pig' : [376,[250]],
                            'redBird' : [385,[488,501]],
                            'blueBird' : [238,[165,280,344,488,416]],
                            'yellowBird' : [497,[288,328]],
                            'whiteBird' : [490,[508,510]],
                            'blackBird' : [488,[146,64,0]],
                            'stone' : [365,[292,438,146]],
                            'ice' : [311,[247]],
                            'wood' : [481,[408,417]],
                            'roundWood' : [490,[344]],
                            'TNT' : [410,[418]],
                            'slingshot' : [346,[136]],
                            'terrain' : [72,[136]]
                             }
#        #quantize to 3-bits
#        for i in range(self._nHeight):
#            for j in range(self._nWidth):
#                self._scene[i,j] = self._quantize(self.screenshot[i,j][0],
#                                                   self.screenshot[i,j][1],
#                                                   self.screenshot[i,j][2])

        #quantize to 3-bits parallel implemtation

        mask = np.ones((self._nHeight,self._nWidth,3)).astype(int) * 0b11100000
        mask = self.screenshot & mask
        mask[:,:,0] = mask[:,:,0] << 1
        mask[:,:,1] = mask[:,:,1] >> 2
        mask[:,:,2] = mask[:,:,2] >> 5
        self._scene = mask[:,:,0] | mask[:,:,1] | mask[:,:,2]

        #segmentation using 8-points
        self._segments = measure.label(self._scene, connectivity=2)
        self._nSegments = np.max(self._segments)+1
        self._colours = np.zeros(self._nSegments)

        #record the value for each segement
        for y in range(self._nHeight):
            for x in range(self._nWidth):
                self._colours[self._segments[y,x]] = self._scene[y,x]

        # we may not need to find the box for all segment at this point

#        #finding a bounding box for each segement
#        for seg in range(self._nSegments):
#            region = np.where(self._segments == seg)
#            rect = Rectangle(region)
#            self._boxes.append(rect)

    def find_pigs(self):
        return self.find_pigs_mbr()

    def find_slingshot_mbr(self):
		#check if the colours of the adjacent points of p is
		#belong to slingshot
        possible_adj_color = [345,418,346,351,281,282,136] #345,418,273,281,209,346,354,282,351,64
#        possible_adj_color = [345,418,346,351,64 ] #345,418,273,281,209,346,354,282,351,64

        box_list = []
        for color in possible_adj_color:

            possible_regions = self._find_box_for_color(color,'slingshot')

            for box in possible_regions:
                #print(box.bottom_right[1])
                #pritn(box.bottom_right[0])
                if box.bottom_right[1] > 200 and box.bottom_right[0] < 200: #this slingshot only appears in the left bottom part
                    box_list.append(box)

        #merge the boxes
        go = 1
        while go:
            new_list=[]
            ignore_list=[]
            go = 0
            for i in range(len(box_list)):
                if i not in ignore_list:
                    for j in range(i+1,len(box_list)):
                        if j not in ignore_list:
                        #print(box_list[i].top_left)
                            if box_list[i].intersects(box_list[j]):
                                go=1
                                box_list[i].add(box_list[j])
                                ignore_list.append(j)
            #print(go)
            if go == 0:
                break

            for i in range(len(box_list)):
                if i not in ignore_list:
                    new_list.append(box_list[i])

            box_list = new_list

        #return the largest rectangle
        max_area = 0
        hw_ratio = 0
        ret = []
        #print(len(box_list))
        for box in box_list:

            area = box.width * box.height
            #print('area:',area)
            if area != 0:
                hw_ratio = box.height/box.width
                #print('ratio:',hw_ratio)
                if area > max_area and hw_ratio > 1.5:
                    ret = box
                    max_area = area
        #self._plot_bounding_box([ret])
        return {'slingshot':[ret]}

    def find_bird_on_sling(self,birds,sling):

        sling_top_left = sling.top_left[1]
        distance = {}
        for bird_type in birds:
            if len(birds[bird_type]) > 0:
                for bird in birds[bird_type]:
                    #print(bird)
                    distance[bird] = abs(bird.top_left[1]\
                                    - sling_top_left)

        min_distance = 1000
        for bird in distance:
            if distance[bird] < min_distance:
                ret = bird
                min_distance = distance[bird]

        return ret

    #the general idea to find block is to use a specific color to find each of the
    #stone, wood, and ice, and then check for adjacent points, if the number of
    #adjacent points that have the same color is larger than certain thershold
    #we say we have found our object.
    def find_tnts_mbr(self):
        main_color = self._color_values['TNT'][0]
        associate_color = self._color_values['TNT'][1]
        return self._find_objects(main_color, associate_color, 'TNT')

    def find_round_woods_mbr(self):
        main_color = self._color_values['roundWood'][0]
        associate_color = self._color_values['roundWood'][1]
        return self._find_objects(main_color, associate_color, 'roundWood')

    def find_ices_mbr(self):
        main_color = self._color_values['ice'][0]
        associate_color = self._color_values['ice'][1]
        return self._find_objects(main_color, associate_color, 'ice')

    def find_stones_mbr(self):
        main_color = self._color_values['stone'][0]
        associate_color = self._color_values['stone'][1]
        return self._find_objects(main_color, associate_color, 'stone')

    def find_woods_mbr(self):
        main_color = self._color_values['wood'][0]
        associate_color = self._color_values['wood'][1]
        return self._find_objects(main_color, associate_color, 'wood')

    def find_terrain_mbr(self):
        #print(len(box_list))
        possible_adj_color = [136] #72
        box_list = []
        for color in possible_adj_color:

            possible_regions = self._find_box_for_color(color,'terrain')

            for box in possible_regions:
                    box_list.append(box)

        go = 1
        while go:
            new_list=[]
            ignore_list=[]
            go = 0
            for i in range(len(box_list)):
                if i not in ignore_list:
                    for j in range(i+1,len(box_list)):
                        if j not in ignore_list:
                        #print(box_list[i].top_left)
                            if box_list[i].intersects(box_list[j]):
                                go=1
                                box_list[i].add(box_list[j])
                                ignore_list.append(j)



            for i in range(len(box_list)):
                if i not in ignore_list:
                    new_list.append(box_list[i])

            box_list = new_list

        merged_box = self._check_val_color(box_list,1,'terrain')
        return {"terrain":set(merged_box)}





    # the general idea to find birds and pigs is to use a specifical color that
    # is unique to each object and dialate the bounding box with checking

    # specific secondray color
    def find_birds(self):
        red = self.find_red_birds_mbr()
        yellow = self.find_yellow_birds_mbr()
        blue = self.find_blue_birds_mbr()
        white = self.find_white_birds_mbr()
        black = self.find_black_birds_mbr()
        birds = [red,yellow,blue,white,black]
        ret = {}
        for bird in birds:
            if len(bird) != 0:
                ret.update(bird)
        return ret


    def find_blocks(self):
        wood = self.find_woods_mbr()
        roundWood = self.find_round_woods_mbr()
        ice = self.find_ices_mbr()
        stone = self.find_stones_mbr()
        terrain = self.find_terrain_mbr()
        blocks = [wood,roundWood,ice,stone,terrain]
        ret = {}
        for block in blocks:
            if len(block) != 0:
                ret.update(block)
        return ret

    def find_pigs_mbr(self):
        '''
        find all the pigs in screenshot
        returns a dictionary as {"pig":pig_objects_list}
        '''

        main_color = self._color_values['pig'][0]
        associate_color = self._color_values['pig'][1]

        return self._find_objects(main_color, associate_color, 'pig')


    def find_red_birds_mbr(self):
        main_color = self._color_values['redBird'][0]
        associate_color = self._color_values['redBird'][1]

        return self._find_objects(main_color, associate_color, 'redBird')

    def find_blue_birds_mbr(self):
        main_color = self._color_values['blueBird'][0]
        associate_color = self._color_values['blueBird'][1]

        return self._find_objects(main_color, associate_color, 'blueBird')

    def find_yellow_birds_mbr(self):
        main_color = self._color_values['yellowBird'][0]
        associate_color = self._color_values['yellowBird'][1]
        return self._find_objects(main_color, associate_color, 'yellowBird')

    def find_white_birds_mbr(self):
        main_color = self._color_values['whiteBird'][0]
        associate_color = self._color_values['whiteBird'][1]

        return self._find_objects(main_color, associate_color, 'whiteBird')

    def find_black_birds_mbr(self):
        main_color = self._color_values['blackBird'][0]
        associate_color = self._color_values['blackBird'][1]

        return self._find_objects(main_color, associate_color, 'blackBird')

    def _find_box_for_color(self,color,object_type):
        '''
        input the color and dialate effect value
        find the segements that corrspond to a particular color
        return the bounding boxes the segements
        '''

        box_list = []
        for seg in np.where(self._colours == color)[0]:
            region = np.where(self._segments == seg)
            box = Rectangle(region)


            if object_type == 'pig' or object_type == 'blackBird':

                box.dialate(dx = box.width /2 + 1,
                            dy = box.height /2 + 1)

            elif object_type == 'redBird':
                box.dialate(dx = 1,
                            dy = box.height /2 + 1)

            elif object_type == 'blueBird':
                box.dialate(dx = 2,
                            dy = box.height /2 + 1)

            elif object_type == 'whiteBird':
                box.dialate(dx = 2,
                            dy = 4)

            elif object_type in ('yellowBird','whiteBird','TNT') :
                box.dialate(dx = 2,
                            dy = 2)

            elif object_type == 'ice':
                box.dialate(dx = 1,
                            dy = 1)

            elif object_type == 'terrain':
                box.dialate(dx = 1,
                            dy = 1)
            else:
                box.dialate(dx = 1,
                            dy = 1)
            game_object = GameObject(box, GameObjectType(object_type))
            box_list.append(game_object)

        return box_list

    def _check_intersact_and_merge(self,box_list):
        '''
        check instersacting boxes and merge them
        return the merged box_list
        '''
        merged_box = []

        for box in box_list:
            if len(merged_box) == 0:
                box.check_val(self._nWidth, self._nHeight)
                merged_box.append(box)
            else:
                if merged_box[-1].intersects(box):
                    merged_box[-1].add(box)
                else:
                    box.check_val(self._nWidth, self._nHeight)
                    merged_box.append(box)

        return merged_box

    def _check_val_color(self,merged_box,colors,object_type):
        '''
        input a list of colors
        check if the object has certain colors inside the bounding box
        return the removed boxes that have at least one of the colors
        '''
        ret_list = set()
        if object_type == 'blackBird':
            for box in merged_box:
                #calculate the number of colors in the segment
                color_count = self._count_color(box, colors)
                #print(color_count)
                try:
                    if 0 in color_count and color_count[0] > max(32,0.1 * box.width * box.height) and \
                        color_count[64] > 0 and 385 not in color_count:
                            ret_list.add(box)
                            
                except:
                    print("color 64 is not in color count")

        elif object_type == 'whiteBird':
            for box in merged_box:
                #calculate the number of colors in the segment
                if box.top_left[1] > 100:
                    color_count = self._count_color(box, colors)
                    if box.width*box.height > self._regionTreshold:
                        if 510 in color_count and 508 in color_count:
                            ret_list.add(box)

        elif object_type == 'yellowBird':
            for box in merged_box:
                #calculate the number of colors in the segment
                box_seg = self._segments[box.top_left[1]:box.bottom_right[1],
                                         box.top_left[0]:box.bottom_right[0]]
                #check color value for each segment
                for seg in np.unique(box_seg):
                    #print(self._colours[seg])
                    #print(colors)
                    if self._colours[seg] in colors:
                        ret_list.add(box)

        elif object_type == 'stone':
            for box in merged_box:
                #calculate the number of colors in the segment
                color_count = self._count_color(box, colors)
                #print(color_count)
                if 292 in color_count and 438 in color_count and 146 in color_count and 508 not in color_count:
                    #print(len(box.points[0]))
                    if box.width*box.height > self._regionTreshold:
                        if box.top_left[0]>100:
                            ret_list.add(box)

        elif object_type == 'wood':
            for box in merged_box:
                #calculate the number of colors in the segment
                color_count = self._count_color(box, colors)
                #print(color_count)

                if 481 in color_count and 408 in color_count and 417 in color_count:
                    #print(len(box.points[0]))

                    if box.width*box.height > self._regionTreshold:
                        ret_list.add(box)

        elif object_type == 'TNT':
            for box in merged_box:
                #calculate the number of colors in the segment
                color_count = self._count_color(box, colors)
                #print(color_count)

                if 481 in color_count and 457 in color_count and 511 in color_count:
                    #print(len(box.points[0]))
                    if box.width*box.height > self._regionTreshold:
                        ret_list.add(box)

        elif object_type == 'terrain':
            for box in merged_box:
                #calculate the number of colors in the segment
                color_count = self._count_color(box, colors)

                if 136 in color_count and 72 in color_count and 282 in color_count and 346 in color_count \
                and 511 in color_count and 153 not in color_count and 418 not in color_count:
                    #print(len(box.points[0]))

                    if box.width*box.height > self._regionTreshold:
                        ret_list.add(box)

        elif object_type == 'slingshot':
            for box in merged_box:
                #calculate the number of colors in the segment
                color_count = self._count_color(box, colors)

                if 136 in color_count and 345 in color_count \
                and 418 in color_count and 282 in color_count \
                and 209 in color_count :
                    #print(len(box.points[0]))
                    if len(box.points[0]) > self._regionTreshold:
                        ret_list.add(box)

        elif object_type == 'pig':
            for box in merged_box:
                #find the segments in the box
                box_seg = self._segments[box.top_left[1]:box.bottom_right[1],
                                         box.top_left[0]:box.bottom_right[0]]
                #check color value for each segment
                for seg in np.unique(box_seg):
                    if self._colours[seg] in colors:
                        if box.width*box.height > 5:
                            ret_list.add(box)
        else:
            for box in merged_box:
                #find the segments in the box
                box_seg = self._segments[box.top_left[1]:box.bottom_right[1],
                                         box.top_left[0]:box.bottom_right[0]]
                #check color value for each segment
                for seg in np.unique(box_seg):

                    if self._colours[seg] in colors:
                        if box.width*box.height > self._regionTreshold:
                            if object_type == 'roundWood':
                                if abs(box.width-box.height) < box.width*0.2: #make sure it is sqaure
                                    ret_list.add(box)
                            else:
                                ret_list.add(box)

        return ret_list

    def _count_color(self,box,colors):
        '''
        count the number of colors in the box
        return dict {color:counts}
        '''
        box_seg = self._segments[box.top_left[1]:box.bottom_right[1],
                                 box.top_left[0]:box.bottom_right[0]]
        color_count = {}

        for i in range(box_seg.shape[0]):
            for j in range(box_seg.shape[1]):
                try:
                    color_count[int(self._colours[box_seg[i,j]])] += 1
                except KeyError:
                    color_count[int(self._colours[box_seg[i,j]])] = 1

        return color_count

    def _quantize(self,r,g,b):
        '''
        no need anymore
        '''
        return (r & 0b11100000) << 1 | \
               (g & 0b11100000) >> 2 | \
               (b & 0b11100000) >> 5



    def _find_objects(self, main_color, associate_color, object_type):
        box_list = self._find_box_for_color(main_color,object_type)
        merged_box = self._check_intersact_and_merge(box_list)
        merged_box = self._check_val_color(merged_box,associate_color,object_type)
        #self._plot_bounding_box(merged_box)

        return {object_type : merged_box}

    def _int2img(array):
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

    def _plot_bounding_box(self, boxes_list):
        #plt
        self.screenshot = cv2.cvtColor(self.screenshot, cv2.COLOR_RGB2BGR)
        for box in boxes_list:
            cv2.rectangle(self.screenshot,
                          tuple(box.top_left),
                          tuple(box.bottom_right), (255,0,0), 1)
        cv2.imshow('res',self.screenshot)
        cv2.waitKey(30)
        cv2.destroyAllWindows()



##for test purpose
if __name__ == "__main__":

#    path = os.listdir('./test_screenshot/')
#    path = [p for p in path if p[:6]=='screen']
#    for p in path:
    img = cv2.imread('./screenshot.png')
    #cv2.imwrite('/Users/chengxue/Desktop/screenshot_cv2.png',img[:,:,1])
    #plt.imshow(img)
    #plt.show()
    tt=time.time()
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = time.time()
    vision_mbr = VisionMBR(img)
    print(time.time()-t)


    t = time.time()

    sling = vision_mbr.find_slingshot_mbr()
    print("slingshow:" + str(time.time()-t))
    
#    t = time.time()
#    pigs = vision_mbr.find_pigs()
#    print(time.time()-t)

#        bird_on_sling = vision_mbr.findBirdOnSlingMBR(birds,sling)
    
#        print('total time required: ' + str(time.time()-tt) )

    objects = [sling]
    objects_dict = {}
    for obj in objects:
        objects_dict.update(obj)
    print (objects_dict)
    for object_type in objects_dict:
        for obj in objects_dict[object_type]:
            cv2.rectangle(img, tuple(obj.top_left),tuple(obj.bottom_right),(255,0,0),1)
            cv2.putText(img,object_type,tuple(obj.top_left),0,0.3,(0,0,255))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('res',img)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
