#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 14:47:00 2020

@author: cheng
"""

import cv2 
import numpy as np
import json
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation
import sys
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


class batch_gt_viewer:
    def __init__(self):

        self.color_dict = {
            'Platform': [25, 0, 51], # purple
            'slingshot': [51, 25, 0], #brown
            'bird_red': [255, 0, 0],
            'bird_yellow' : [255, 255, 0], #yellow
            'bird_blue': [0, 0, 255],
            'bird_black':[0, 0, 0],
            'bird_white': [224, 224, 224],
            'pig':[0, 255, 0],
            'ice':[0, 255, 255],
            'wood': [255, 128, 0],
            'stone': [64, 64, 64],
            'TNT':[207, 83, 0],
            'Object':[255, 28, 0],
            'unknown': [255,255,255]
            }

                


    def show(self, path):
        
        with open(path) as f:
            
            self.batch_gt = json.load(f)
    

        #save object id: [type, coodinates: []]
        self.ret = {}
        
        #assuming all objects are visiable in the first gt
        for obj in self.batch_gt[0][0]['features']:
        
            obj_id = obj['properties']['id']
            
            if obj['properties']['label'] != 'Ground' and obj['properties']['label'] != "Trajectory" and obj['properties']['label'] != "Slingshot":
                if obj['properties']['label'] != 'TNT' and obj['properties']['label'] != 'Platform' and obj['properties']['label'] != 'Object':
                    label = "_".join(obj['properties']['label'].split("_")[:-1])
                else:
                    label = obj['properties']['label']
                
                self.ret[obj_id] = {'label':label, 'coordinates':[]}
        
        
        for i, gt in enumerate(self.batch_gt):
            gt = gt[0]['features']
            
        
            for obj_id in self.ret:
                updated = False        
                for obj in gt:
                    if obj['properties']['label'] == 'Ground':
                        self.ground = obj['properties']["yindex"]                    
                    obj_id_gt = obj['properties']['id']
                    
                    if obj_id == obj_id_gt:

                        if obj['properties']['label'] != 'Ground' and obj['properties']['label'] != "Trajectory" and obj['properties']['label'] != "Slingshot":
                            if obj['properties']['id'] not in self.ret:
                                self.ret[obj['properties']['id']] = {}
                                self.ret[obj['properties']['id']]['label'] = obj['properties']['label']
                                coods = np.array(obj['geometry']['coordinates'][0])
                                #coods[:,1] -= 480
                                self.ret[obj['properties']['id']]['coordinates'] = [coods]
                            else:
                                coods = np.array(obj['geometry']['coordinates'][0])
                                #coods[:,1] -= 480
                                self.ret[obj['properties']['id']]['coordinates'].append(coods)
                            updated = True
                
                if updated == False:
                    if 'bird' in self.ret[obj_id]['label']:
                        print(self.ret[obj_id]['label'])
                        print(obj_id)
                        print('missing')
                        print(i)
                        #print(gt)
                        #sys.exit()
                    self.ret[obj_id]['coordinates'].append(np.zeros_like(self.ret[obj_id]['coordinates'][-1]))
                    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0,900)
        ax.set_ylim(-100,480)
        ax.axhline(self.ground)
        plt.gca().invert_yaxis()
        patches = {}
        
        for obj_id in self.ret:
            polygon = Polygon( self.ret[obj_id]['coordinates'][0], True)


            col = list(filter(lambda x : x in self.ret[obj_id]['label'], self.color_dict.keys()))[0]
                    
            
            polygon.set_color(np.array(self.color_dict[col])/255)
            patches[obj_id] = polygon
            
        for patch in patches:
           ax.add_patch(patches[patch])
        
        #plt.show()
        
        #patch = patches.Polygon(v,closed=True, fc='r', ec='r')
        #ax.add_collection(p)
        
        def init():
            for patch in patches:
               ax.add_patch(patches[patch])
            return patches.values()
        
        def animate(ind):
            for patch in patches:
                coords = self.ret[patch]['coordinates'][ind]
                patches[patch].set_xy(coords)
            return patches.values()
        
        ani = animation.FuncAnimation(fig, animate, np.arange(1, len(self.batch_gt)), init_func=init,
                                      interval=500, blit=True)
        plt.show()


if __name__ == '__main__':
    path = 'batch_gt.json'
    viewer = batch_gt_viewer()
    viewer.show(path)
