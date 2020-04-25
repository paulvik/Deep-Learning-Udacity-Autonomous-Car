
import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import tensorflow as tf



def create_target_set(driving_log_dir):
    target_steering = []
    targets = pd.read_csv(driving_log_dir, sep=",", header=None)
    targets = targets.values
    
    for images in range(len(targets)):
        target_steering.append(targets[images][3])
    
    return target_steering


def split_data_set_by_camera(image_list):

    center_camera_temp = []
    left_camera_temp = []
    right_camera_temp = []

    for image in range(len(image_list)):
    
        image_split = image_list[image].split("_")

        if image_split[0] == "center":
            center_camera_temp.append(image_list[image])

        elif image_split[0] == "left":
            left_camera_temp.append(image_list[image])

        elif image_split[0] == "right":
            right_camera_temp.append(image_list[image])

    return center_camera_temp, left_camera_temp, right_camera_temp


def crop_images(image_list, directory):
    cropped_images = []
    
    for image in range(len(image_list)):
        img = Image.open(directory + "/" + image_list[image])
        width, height = img.size
        cropped_image = img.crop((0, 65, width, height - 20))
        cropped_images.append(cropped_image)
        cropped_image.save(directory + "_CROP/" + image_list[image], 'jpeg')

    return cropped_images


def create_input_vector(image_dir):
    image = Image.open(image_dir)
    image_array = np.asarray(image)
    image_array = image_array[..., ::-1]
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
    image_array = (image_array / 255) - 0.5
    return image_array



def create_input_vector_with_flip(image_dir):
    image = cv2.imread(image_dir)
    image_flip = cv2.flip(image, 0)
    image_array = np.asarray(image_flip)
    image_array = image_array[..., ::-1]
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
    image_array = (image_array / 255) - 0.5
    return image_array

     
        
def create_data_set(image_list, driving_log_dir, image_dir):
    # This function should append the images from the center, left and right camera into one dataset and adjust the steering angle such
    # that the angle of each camera corresponds to the center camera. 
    # Returns an array where each row contains an image with its corresponding corrected target value.    
    total_data_set = []
    
    center_camera, left_camera, right_camera = split_data_set_by_camera(image_list)
    
    center_target = create_target_set(driving_log_dir)
    left_target = create_target_set(driving_log_dir)
    right_target = create_target_set(driving_log_dir)
    
    # Adjusting the angles    
    for value in range(len(left_target)):
        left_target[value] += 0.25
        right_target[value] -= 0.25
    
    
    for image in range(len(center_camera)):
        total_data_set.append([create_input_vector(image_dir + "/" + center_camera[image]), center_target[image]])
        total_data_set.append([create_input_vector(image_dir + "/" + left_camera[image]), left_target[image]])
        total_data_set.append([create_input_vector(image_dir + "/" + right_camera[image]), right_target[image]])
            
    return total_data_set

   
    
def split_data_set(data_set):
    images = []
    targets = []
    
    for element in range(len(data_set)):
        images.append(data_set[element][0])
        targets.append(data_set[element][1])
    
    return images, targets



        

    
        
        
    
    
    
    



