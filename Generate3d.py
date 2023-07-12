import torch 
import cv2
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt
import time 
import bpy 
import math
from ObjectDetect import ObjectDetection


color_mapping_blender =  {
    'room': (0.0, 0.0, 1.0, 1.0),        # Blue
    'door': (1.0, 0.0, 0.0, 1.0),        # Red
    'window': (0.0, 1.0, 0.0, 1.0),      # Green
    'wc': (1.0, 0.647, 0.0, 1.0),        # Orange
    'toilet': (0.502, 0.0, 0.502, 1.0),  # Purple
    'sofa': (1.0, 1.0, 0.0, 1.0),        # Yellow
    'kitchen': (0.0, 1.0, 1.0, 1.0),     # Cyan
    'bed': (1.0, 0.412, 0.706, 1.0),     # Pink
    'storage': (0.165, 0.165, 0.647, 1.0),  # Brown
    'tv': (0.502, 0.502, 0.502, 1.0),    # Gray
    'dining_table': (1.0, 0.0, 1.0, 1.0)  # Magenta
}


class Generate3d:
    
    def __init__(self,img_path,target_path):
        self.img_path = img_path 
        self.target_path = target_path
        
    def GetPredtions(self):
        wallDetect = ObjectDetection('walldetector',img_path=self.img_path)
        self.wall_predictions = wallDetect.Predtict()
        roomDetect = ObjectDetection('RoomDetector',self.img_path)
        self.room_predictions = roomDetect.Predtict()
        windowDetect = ObjectDetection('windowDetector',self.img_path)
        self.window_predictions = windowDetect.Predtict()
        doorDetect = ObjectDetection('DoorDetector',img_path=self.img_path)
        self.door_predictions = doorDetect.Predtict()
        
    def place_model(self,obj_filepath, position,rotation,dimension):
        bpy.ops.import_scene.obj(filepath=obj_filepath)
        bed = bpy.context.selected_objects[0]
        global_coords = bed.matrix_world.translation
        print("Global Coordinates:", global_coords)

        # Get the object's local coordinates
        local_coords = bed.location
        print("Local Coordinates:", local_coords)
        # Get the object's global scale
        global_scale = bed.matrix_world.to_scale()
        print("Global Scale:", global_scale)
        # position = ( position_x,position_y,posisiton_z)
        # dimension = (rotation_x,rotation_y,rotation_z)
        # rotation  = (dimension_x,dimension_y,dimension_z)
        # Get the object's local scale
        local_scale = bed.scale
        print("Local Scale:", local_scale)
        bed.dimensions.xyz = dimension
        bed.location = position
        rotation_radians = [math.radians(angle) for angle in rotation]
        bed.rotation_euler = rotation_radians
        
    def GenerateModel(self):
        self.GetPredtions()
        # Clear existing objects in the scene
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type='MESH')
        bpy.ops.object.delete()
        # Create a cube object for each bounding box for walls 
        for bbox in self.wall_predictions['predictions']:
            x = bbox['x']
            y = bbox['y']
            width = bbox['width']
            height = bbox['height']
            obj_class = bbox['class']
            # Calculate the coordinates of the bounding box corners
            # x1 = x
            # y1 = y
            # x2 = x + width
            # y2 = y + height
            # x0 = bbox['x'] - bbox['width'] / 2
            # x1 = bbox['x'] + bbox['width'] / 2
            # y0 = bbox['y'] - bbox['height'] / 2
            # y1 = bbox['y'] + bbox['height'] / 2
            
            # Create a cube mesh
            bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
            cube = bpy.context.object
            
            # Scale the cube to match the bounding box dimensions
            # cube.dimensions.x = 100
            # cube.dimensions.y = 50
            # cube.dimensions.z = 150
            cube.dimensions.xyz = [width,height,150]
            
            # Position the cube at the center of the bounding box
            cube.location.x = x
            cube.location.y = y 
            cube.location.z = -150/2  # Set the desired height of the bounding box
            
            # Assign a material to the cube
            material = bpy.data.materials.new(name="BoundingBoxMaterial")
            
            material.diffuse_color = (0.502, 0.502, 0.502, 1.0)  
        cube.data.materials.append(material)
        
        for bbox in self.window_predictions['predictions']:
            x = bbox['x']
            y = bbox['y']
            width = bbox['width']
            height = bbox['height']
            obj_class = bbox['class']
            # Calculate the coordinates of the bounding box corners
            # x1 = x
            # y1 = y
            # x2 = x + width
            # y2 = y + height
            # x0 = bbox['x'] - bbox['width'] / 2
            # x1 = bbox['x'] + bbox['width'] / 2
            # y0 = bbox['y'] - bbox['height'] / 2
            # y1 = bbox['y'] + bbox['height'] / 2
            
            # Create a cube mesh
            bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
            cube = bpy.context.object
            
            # Scale the cube to match the bounding box dimensions
            # cube.dimensions.x = 100 
            # cube.dimensions.y = 50
            # cube.dimensions.z = 150 
            cube.dimensions.xyz = [width,height,150]
            
            # Position the cube at the center of the bounding box
            cube.location.x = x
            cube.location.y = y 
            cube.location.z = -150/2  # Set the desired height of the bounding box
            
            # Assign a material to the cube
            material = bpy.data.materials.new(name="BoundingBoxMaterial")
            if obj_class == 'window':
                material.diffuse_color = (0.502, 0.502, 0.502, 1.0) # red color for windows
                # Create a hole in the cube for windows using a Boolean modifier
                bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
                hole_cube = bpy.context.object
                # hole_cube.dimensions.x = width -10
                # hole_cube.dimensions.y = height +50
                # hole_cube.dimensions.z = 50  # Increase the height of the hole cube
                hole_cube.dimensions.xyz = [width-10,height+50,50]
                hole_cube.location.x = x
                hole_cube.location.y = y
                hole_cube.location.z = -150/2 # Set the height above the main cube
                cube.modifiers.new(name="Boolean", type='BOOLEAN')
                cube.modifiers['Boolean'].operation = 'DIFFERENCE'
                cube.modifiers['Boolean'].object = hole_cube
                # cube.modifiers['Boolean'].overlap_threshold = 0.0
                hole_cube.hide_viewport = True  # Hide the hole cube from the viewport
                hole_cube.hide_render = True
                
            else:
                material.diffuse_color = (0.502, 0.502, 0.502, 1.0)  
        cube.data.materials.append(material)
        
        for bbox in self.room_predictions['predictions']:
            x = bbox['x']
            y = bbox['y']
            width = bbox['width']
            height = bbox['height']
            obj_class = bbox['class']
            # Calculate the coordinates of the bounding box corners
            # x1 = x
            # y1 = y
            # x2 = x + width
            # y2 = y + height
            # x0 = bbox['x'] - bbox['width'] / 2
            # x1 = bbox['x'] + bbox['width'] / 2
            # y0 = bbox['y'] - bbox['height'] / 2
            # y1 = bbox['y'] + bbox['height'] / 2
            
            # Create a cube mesh
            bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
            cube = bpy.context.object
            
            # Scale the cube to match the bounding box dimensions
            # Set the desired height of the bounding box
            
            # Assign a material to the cube
            material = bpy.data.materials.new(name="BoundingBoxMaterial")
            if obj_class == 'room' or obj_class == 'wc':
                # cube.dimensions.x = width 
                # cube.dimensions.y = height
                # cube.dimensions.z = 0 
                cube.dimensions.xyz = [width,height,0]
                
                # Position the cube at the center of the bounding box
                cube.location.x = x
                cube.location.y = y 
                cube.location.z = 0 
                material.diffuse_color = color_mapping_blender[obj_class]
                
                
                
            else:
                # cube.dimensions.x = width 
                # cube.dimensions.y = height
                # cube.dimensions.z = -20
                cube.dimensions.xyz = [width,height,-20]
                
                # Position the cube at the center of the bounding box
                cube.location.x = x
                cube.location.y = y 
                cube.location.z = -10
                bed_path = r"C:/capstone/3dFloorplanCopy/Furniture/bed/ikea_malm_obj/ikea_malm_obj.obj"
                if obj_class == 'bed':
                    self.place_model( bed_path, position=(x, y, -10), rotation=(-90, 0, -360), dimension= [width,10,height])
                
                # elif obj_class == 'toilet':
                    # self.place_model(r'C:\capstone\3dFloorplan - Copy\Furniture\toilet\Toilet_OBJ\Toilet_OBJ.obj', x, y, -10, 0, 0, -360, width,10,height)
                    
                material.diffuse_color = color_mapping_blender[obj_class] 
                
            cube.data.materials.append(material)
        bpy.ops.wm.save_as_mainfile(filepath = self.target_path)
        
    def euclidean_distance(self,x1,y1,x2,y2):
        return math.sqrt((x1-x2)**2 + (y1-y2)**2) 
        
    def getRooms(self):
        bedrooms = [room for room in self.room_predictions['predictions'] if room['class'] == 'bed']
        return bedrooms
    
    def getCloseWalls(self):
        # returns only for one room cause the return statment is inside the for loop 
        bedrooms = self.getRooms()
        for i in bedrooms:
            room_x = i.get('x')
            room_y = i.get('y')
            width = i.get('width')
            height = i.get('height')
            distances = []
            n =5
            for wall in self.wall_predictions['predictions']:
                distance = self.euclidean_distance(room_x,room_y, wall['x'],wall['y'])
                distances.append((wall, distance))
            distances.sort(key=lambda x: x[1])  # Sort distances in ascending order
            closest_points = [point for point, distance in distances[:n]]
            # for i in closest_points:
            #     print(i['x'])
            return closest_points
    
        
    def AddColor():
        pass
    
    
if __name__ == '__main__':
    img_path = r'C:\capstone\3dFloorplanCopy\Capstone-4\valid\images\16_jpg.rf.21cbc6ecfeec59d54a1af55ead68c60c.jpg'
    
    generate3d = Generate3d(img_path=img_path,target_path=r'C:\capstone\3dFloorplanCopy\output.blend')
    generate3d.GenerateModel()