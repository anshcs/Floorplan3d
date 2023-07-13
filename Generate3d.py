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
            
            
            # Create a cube mesh
            bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
            cube = bpy.context.object
            
            # Scale the cube to match the bounding box dimensions
            
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
            
            
            # Create a cube mesh
            bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
            cube = bpy.context.object
            
            # Scale the cube to match the bounding box dimensions
            
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
               
                hole_cube.dimensions.xyz = [width-10,height+50,50]
                hole_cube.location.x = x
                hole_cube.location.y = y
                hole_cube.location.z = -150/2 # Set the height above the main cube
                cube.modifiers.new(name="Boolean", type='BOOLEAN')
                cube.modifiers['Boolean'].operation = 'DIFFERENCE'
                cube.modifiers['Boolean'].object = hole_cube
                
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
            
            
            # Create a cube mesh
            bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
            cube = bpy.context.object
            
            # Scale the cube to match the bounding box dimensions
            # Set the desired height of the bounding box
            
            # Assign a material to the cube
            material = bpy.data.materials.new(name="BoundingBoxMaterial")
            if obj_class == 'room' or obj_class == 'wc':
                
                cube.dimensions.xyz = [width,height,0]
                
                # Position the cube at the center of the bounding box
                cube.location.x = x
                cube.location.y = y 
                cube.location.z = 0 
                material.diffuse_color = color_mapping_blender[obj_class]
                
                
                
            else:
                
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
            
            return closest_points
    
        
    def AddColor(self):
        filepath = self.target_path
        bedrooms =  self.getRooms()
        room_x = bedrooms[0].get('x')
        room_y = bedrooms[0].get('y')
        closest_points = self.getCloseWalls()
        bpy.ops.wm.open_mainfile(filepath=filepath)

        wallColor = bpy.data.materials.new(name="Blue Material")
        wallColor.diffuse_color = (0.0, 0.0, 1.0, 1.0)
        camera_location = (room_x, room_y,-50)
        # Set the active object to the camera

        camera = bpy.context.scene.camera
        camera.location = camera_location
        camera_location = bpy.context.scene.camera.location
        for i in closest_points:
            # Deselect all objects initially
            bpy.ops.object.select_all(action='DESELECT')
            # irrerate through objects 
            for obj in bpy.data.objects:
                # get objects at specified loc 
                if (obj.location.x == i['x'] and obj.location.y == i['y'] and obj.location.z == -75.00):
                    obj.select_set(True)
                    # get normal vector 
                    view_vector = obj.location - camera_location
                    for face_index, face in enumerate(obj.data.polygons): 
                        normal_vector = face.normal 
                        dot_product = normal_vector.dot(view_vector)
                        # check if camera is inside or outside room 
                        if dot_product <0 :
                            
                            wallColor = bpy.data.materials.new(name="Blue Material")
                            
                            obj.data.materials.append(wallColor)
                            obj.active_material = wallColor
                            obj.data.materials.append(bpy.data.materials.new(name="FaceMaterial"))
                            face_color = (0.0, 0.0, 1.0, 1.0)

                            # Assign the face material to the desired face(s)
                           
                            obj.data.polygons[face_index].material_index = 1

                            # # Set the color of the face material
                            obj.data.materials[1].diffuse_color = face_color
                    

                

                    # Switch back to Object mode
                    bpy.ops.object.mode_set(mode='OBJECT')
            
        bpy.ops.wm.save_as_mainfile(filepath=r'C:\capstone\3dFloorplanCopy\final.blend')
    def AddTexture(self):
        filepath = self.target_path
        bedrooms =  self.getRooms()
        room_x = bedrooms[0].get('x')
        room_y = bedrooms[0].get('y')
        closest_points = self.getCloseWalls()
        bpy.ops.wm.open_mainfile(filepath=filepath)

        wallColor = bpy.data.materials.new(name="Blue Material")
        wallColor.diffuse_color = (0.0, 0.0, 1.0, 1.0)
        camera_location = (room_x, room_y,-50)
        # Set the active object to the camera

        camera = bpy.context.scene.camera
        camera.location = camera_location
        camera_location = bpy.context.scene.camera.location
        for i in closest_points:
            # Deselect all objects initially
            bpy.ops.object.select_all(action='DESELECT')
            for obj in bpy.data.objects:
                if (obj.location.x == i['x'] and obj.location.y == i['y'] and obj.location.z == -75.00):
                    obj.select_set(True)
                    view_vector = obj.location - camera_location
                    for face_index, face in enumerate(obj.data.polygons): 
                        normal_vector = face.normal 
                        dot_product = normal_vector.dot(view_vector)
                        if dot_product > 0 :
                            
                            
                            # -----------------------------------------------------------------------------------------------------------
                            obj.data.polygons[face_index].material_index = 1
                            mat = bpy.data.materials.new(name="Material")
                            # wallColor.diffuse_color = (0, 0, 0, 1) 
                            # face.material_index = len(obj.data.materials) - 1
                            obj.data.materials.append(mat)
                            # Create a new texture
                            tex = bpy.data.textures.new(name="MyTexture", type='IMAGE')

                            # Load an image file
                            tex.image = bpy.data.images.load(r'C:\capstone\3dFloorplanCopy\Furniture\textures\texture-wall-background.jpg')  # Replace with your image file path

                            # Create a new material node tree
                            mat.use_nodes = True
                            nodes = mat.node_tree.nodes
                            links = mat.node_tree.links

                            # Clear existing nodes
                            for node in nodes:
                                nodes.remove(node)

                            # Create a texture node
                            tex_node = nodes.new('ShaderNodeTexImage')
                            tex_node.image = tex.image

                            # Create a diffuse BSDF node
                            bsdf_node = nodes.new('ShaderNodeBsdfDiffuse')

                            # Create an output node
                            output_node = nodes.new('ShaderNodeOutputMaterial')

                            # Link nodes
                            links.new(tex_node.outputs[0], bsdf_node.inputs[0])
                            links.new(bsdf_node.outputs[0], output_node.inputs[0])

                            # Set the texture mapping coordinates
                            uv_map_node = nodes.new('ShaderNodeUVMap')
                            uv_map_node.uv_map = "UVMap"  # Replace with the UV map name if necessary

                            links.new(uv_map_node.outputs[0], tex_node.inputs[0])
                            obj.data.uv_layers.new()
                            uv_map = obj.data.uv_layers["UVMap"]
                            for loop_index in face.loop_indices:
                                uv_map.data[loop_index].uv = (0.0, 0.0)



                    # Switch back to Object mode
                    bpy.ops.object.mode_set(mode='OBJECT')



            
        bpy.ops.wm.save_as_mainfile(filepath=r'C:\capstone\3dFloorplanCopy\final2.blend')
    
    
if __name__ == '__main__':
    img_path = r'C:\capstone\3dFloorplanCopy\Capstone-4\valid\images\73_jpg.rf.ddaf8f5db3e713de7e3118d11bf3a327.jpg'
    
    generate3d = Generate3d(img_path=img_path,target_path=r'C:\capstone\3dFloorplanCopy\image73.blend')
    generate3d.GenerateModel()
    generate3d.AddColor()
    generate3d.AddTexture()