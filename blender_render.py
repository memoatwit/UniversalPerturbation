import bpy
import os
import math
import mathutils
import random 
from PIL import Image
import sys

class BlenderRenderer:
    def __init__(self, blend_file_path, texture_path, model_name, output_folder, camera_distance, angle_number):
        self.blend_file_path = blend_file_path
        self.texture_path = texture_path
        self.model_name = model_name
        self.output_folder = output_folder
        self.camera_distance = camera_distance  # Distance you want the camera to be from the model
        self.angle_number = angle_number


    def render(self):
        # Load the .blend file       
        bpy.ops.wm.open_mainfile(filepath=self.blend_file_path)
        # obj = bpy.context.active_object
        # Resize texture
        image = Image.open(self.texture_path)
        image = image.convert('RGB') #if png, convert to jpg and remove alpha ch

        if image.size != (224,224):
            image = image.convert('RGB') #if png, convert to jpg and remove alpha ch
            new_image = image.resize((224, 224))
            new_name = self.texture_path.split('.')
            new_name = new_name[:-1][0]+"_224.jpg"
            new_image.save(new_name)
            self.texture_path = new_name
            print(f'Texture resized before render to: {new_image.size} at {new_name}')

        # Load the texture image      
        texture_image = bpy.data.images.load(self.texture_path)

        # Create a new material with the texture
        material = bpy.data.materials.new(name="Material")
        material.use_nodes = True
        # print(list(material.node_tree.nodes))
        # material.node_tree.nodes.active = material.node_tree.nodes["Material Output"]
        bsdf = material.node_tree.nodes.get('Principled BSDF')
        texture_node = material.node_tree.nodes.new('ShaderNodeTexImage')
        texture_node.image = texture_image
        material.node_tree.links.new(bsdf.inputs['Base Color'], texture_node.outputs['Color'])
        material.node_tree.links.new(material.node_tree.nodes["Material Output"].inputs["Surface"], bsdf.outputs["BSDF"])

        for obj in bpy.data.objects:
            print(obj.name)
        
        # Assign the material to your model
        model_name = self.model_name  # UPDATE with your model's name
        # model_name = obj.name  # UPDATE with your model's name


        if model_name in bpy.data.objects:
            model = bpy.data.objects[model_name]
            if model.data:
                # Accessing materials only if the object has data
                if hasattr(model.data, 'materials'):
                    # model.data.materials.append(material)
                    model.data.materials[0] = material 
                else:
                    # this would give me an error... 
                    # model.data.materials = [material]  # Set materials list
                    # obj.data.materials.append(material)
                    print(f"Object with name '{model_name}' has no 'materials' property.")
            else:
                print(f"Object with name '{model_name}' has no data.")
        else:
            print(f"Object with name '{model_name}' not found.")

        # Set up rendering settings
        bpy.context.scene.render.engine = 'CYCLES'  # or 'BLENDER_EEVEE'
        bpy.context.scene.render.image_settings.file_format = 'JPEG'
        bpy.context.scene.render.resolution_x = 224  # Adjust as needed
        bpy.context.scene.render.resolution_y = 224  # Adjust as needed
        bpy.context.scene.render.resolution_percentage = 100

        # remove default light    
        bpy.ops.object.select_by_type(type='LIGHT')
        bpy.ops.object.delete(use_global=False)

        # Create new light
        lamp_data = bpy.data.lights.new(name="Lamp", type='POINT')
        lamp_data.energy = 100
        lamp_object = bpy.data.objects.new(name="Lamp", object_data=lamp_data)
        bpy.context.collection.objects.link(lamp_object)
        lamp_object.location = (0, 0, 2)

        # Camera setup
        camera = bpy.data.objects['Camera']  # Ensure a camera is present with this name

        camera.location.z = 0  # Set the camera at the same height as the object's origin

        if 'vase' in self.model_name:
            print("raising the cam for the vase")
            camera.location.z = 25  # vase is big
            camera.location.y *= 25  # vase is big
            lamp_object.location = (0, 0, 25)

        if 'desk' in self.model_name:
            print("raising the cam for the table")
            camera.location.z = 1 

        if 'tractor' in self.model_name:
            print("raising the cam for the tractor")
            camera.location.z = 1  #  is big
        #     # camera.location.y *= 2  # vase is big
        #     lamp_object.location = (0, 0, 2)

        # angle_number = 8
        angle = math.pi / 4
        output_folder = self.output_folder  # UPDATE this path

        for i in range(self.angle_number):
            # Calculate camera position for horizontal orbit at a specific distance
            ang = angle * i
            if 'tractor' in self.model_name:
                if ang % (math.pi/2) < 0.1: #if we are behind the truck, move to a new location
                    ang = random.uniform(0, 1.1)
            #     if i==2 or i == 3 or i == 5:
            #         i = i // 2
            camera.location.x = self.camera_distance * math.cos(ang)
            camera.location.y = self.camera_distance * math.sin(ang)
            print(f"{i}, {ang=}, mod: {ang % (math.pi/2)}, {camera.location.x=}, { camera.location.y=}")
            # add randomness: 15%
            camera.location.z *= random.uniform(0.85, 1.15)
            camera.location.x *= random.uniform(0.85, 1.15)
            camera.location.y *= random.uniform(0.85, 1.15)

            

            # Point the camera towards the model
            bpy.context.view_layer.update()
            direction = mathutils.Vector((0, 0, 0)) - mathutils.Vector((camera.location.x, camera.location.y, camera.location.z))

            # Use the direction to calculate the rotation quaternion and then convert it to euler angles
            camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

            lamp_object.location = (camera.location.x, camera.location.y, camera.location.z)
            lamp_object.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
            
            # Render settings for each angle
            bpy.context.scene.render.filepath = os.path.join(output_folder, f"{output_folder.split('/')[-2]}_angle_{i}.jpg")

            # Render
            # Output is very verbose. Redirect output to log file
            logfile = 'blender_render.log'
            open(logfile, 'a').close()
            old = os.dup(sys.stdout.fileno())
            sys.stdout.flush()
            os.close(sys.stdout.fileno())
            fd = os.open(logfile, os.O_WRONLY)

            # do the rendering
            bpy.ops.render.render(write_still=True)

            # disable output redirection
            os.close(fd)
            os.dup(old)
            os.close(old)
            # bpy.ops.render.render(write_still=True)

        print("Rendering complete.")


if __name__ == "__main__":
    from blender_render import BlenderRenderer

    path = '/home/lambda15/Documents/advarch-reinforcement/blender_tests/'
    nrenders = 10
#lemon
    # blender_renderer = BlenderRenderer (
    # blend_file_path = '/home/lambda15/Documents/advarch-reinforcement/blender_tests/lemon.blend',
    # texture_path = '/home/lambda15/Documents/advarch-reinforcement/blender_tests/highpoly_backup.jpg',
    # model_name = 'highpoly',
    # camera_distance = 5.0,
    # output_folder = '/home/lambda15/Documents/advarch-reinforcement/blender_tests/lemon/',
    # angle_number = 10,
    # )

    #Vase
    # blender_renderer = BlenderRenderer (
    # blend_file_path = path + 'vase.blend',
    # texture_path = path+'vase_M_0_0_591128_Normal.jpg',
    # model_name = 'C_vase_low',
    # camera_distance = 35,
    # output_folder = 'vase/',
    # angle_number = 10,
    # )

# #Tray
#     blender_renderer = BlenderRenderer (
#     blend_file_path = path + 'wood_tray.blend',
#     texture_path = path+'wood_tray_diffuse.png',
#     model_name = 'wood_tray',
#     camera_distance = 1,
#     output_folder = 'tray/',
#     angle_number = nrenders,
#     )

#Tray
    blender_renderer = BlenderRenderer (
    blend_file_path = path + 'tractor.blend',
    texture_path = path+'tractor_texture.jpg',
    model_name = 'tractor',
    camera_distance = 3.0,
    output_folder = 'tractor/',
    angle_number = nrenders,
    )

#baseball
    # blender_renderer = BlenderRenderer (
    # blend_file_path = path+'baseball.blend',
    # texture_path = path+'baseball_ball_01_n.png',
    # model_name = 'polySurface5_lambert1_0',
    # output_folder = path+'baseball/',
    # camera_distance= 2.0,
    # angle_number = 10
    # )

    blender_renderer.render()


    # "Tractor" (https://skfb.ly/6GuHP) by selfie 3D scan is licensed under Creative Commons Attribution (http://creativecommons.org/licenses/by/4.0/).