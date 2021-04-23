import math
import numpy as np


class ConvertCoordinates():
    """
    Class that recieves pixel X Y and depth from camera, Camera to eye offsets.
    Converts detected focus point to eye simulator eye X and Y position
    for left and right eye individually in 3D mode.

    Different options of complexity of data conversion available:

    2D_simple: uses pixel location as fraction of resolution and puts that
    fraction between -30 and 30 degree eye rotation for x and y

    2D: uses pixel location as fraction of resolution and puts that
    fraction between -30 and 30 degree eye rotation for x and y PLUS takes 
    into account depth for prominence calculation

    3D: calculate individual eye rotation to be properly oriented to detected
    object


    coordinate system has xyz 0 at camera. 
    positive directions references from doll heads perspective
     x+ right, y+up. z+ away from head in eye direction

     Distances in mm millimeter

     center of "fake eye ball" is set at 10mm behind screen. 
     can be changed with __eye_center_offset_from_screen variable
    """

    def __init__(self) -> None:
        self.__eyeXR = None    
        self.__eyeYR = None
        self.__eyeXL = None  
        self.__eyeYL = None
        self.__mode = ''
        self.__camera_to_between_eyes_offset_x = 180 
        self.__camera_to_between_eyes_offset_y = -100
        self.__camera_to_between_eyes_offset_z = -70 
        self.__eye_offset_R_x = 31  + self.__camera_to_between_eyes_offset_x 
        self.__eye_offset_L_x = -31 + self.__camera_to_between_eyes_offset_x 
        self.__eye_offset_R_y = self.__camera_to_between_eyes_offset_y 
        self.__eye_offset_L_y = self.__camera_to_between_eyes_offset_y
        self.__eye_center_offset_from_screen = -10 
        self.__camera_resolution = (640,480)
        #field of view[degree] horisontal axis ( x) 64 from datasheet
        #field of view[degree] of view vertical axis ( y) 41 from datasheet
        self.__fov_x = 64/2         
        self.__fov_y = 41/2 


    def __calc_eye_coordinates(self,x,y,z):
        if self.__mode == '3D':
            self.__calc_3D(x,y,z)
        elif self.__mode == '2D':
            self.__calc_2D(x,y,z)
        else:
            self.__calc_2D_simple(x,y,z)

    def __calc_3D(self,x,y,depth):        
        """
        X pixel for detection object to focus on
        Y pixel for detection object to focus on
        depth mm to detected object to focus on

        takes in pixel coordinates and depth and sets eye X,Y
        rotation (degrees from 0,0 straight ahead)
        """

        #Calculate X,Y angle relative to eye references
        pix_per_degree_x = self.__camera_resolution[0]/self.__fov_x 
        degrees_from_left = x / pix_per_degree_x  
        degrees_from_center = degrees_from_left - (self.__fov_x/2)         
        x_distance_from_center_mm = \
            math.sin(math.radians(degrees_from_center)) * depth        
        z_distance_from_center_mm = \
            math.cos(math.radians(degrees_from_center))* depth 
        z_distance_from_center_mm = \
            z_distance_from_center_mm - self.__camera_to_between_eyes_offset_z

        pix_per_degree_y = self.__camera_resolution[1]/self.__fov_y 
        degrees_from_top = y / pix_per_degree_y  
        degrees_from_center = degrees_from_top - (self.__fov_y/2) 
        y_distance_from_center_mm = \
                        math.sin(math.radians(degrees_from_center)) * depth
        

        #x and y coordinates relative to eye positions
        left_eye_x = x_distance_from_center_mm - self.__eye_offset_L_x 
        left_eye_y = y_distance_from_center_mm - self.__eye_offset_L_y 

        right_eye_x = x_distance_from_center_mm - self.__eye_offset_R_x 
        right_eye_y = y_distance_from_center_mm - self.__eye_offset_R_y 
        

        left_eye_x_angle = math.asin(left_eye_x/z_distance_from_center_mm)
        left_eye_y_angle = math.asin(left_eye_y/z_distance_from_center_mm)

        right_eye_x_angle = math.asin(right_eye_x/z_distance_from_center_mm)
        right_eye_y_angle = math.asin(right_eye_y/z_distance_from_center_mm)

                
        self.__eyeXL = math.degrees(left_eye_x_angle)
        self.__eyeYL = -math.degrees(left_eye_y_angle)
        self.__eyeXR = math.degrees(right_eye_x_angle)
        self.__eyeYR = -math.degrees(right_eye_y_angle)    
        
    def print_current_eye_angles(self):
        print(f'Left Eye X: {self.__eyeXL}')
        print(f'Left Eye Y: {self.__eyeYL}')
        print(f'Right Eye X: {self.__eyeXR}')
        print(f'Right Eye X: {self.__eyeYR}')

        
    def __calc_2D(self, x,y,z):
        self.__eyeXR = round(-30 + (x / self.__camera_resolution[0]) * 60, 2)
        self.__eyeYR = round(-30 + (1-(y / self.__camera_resolution[1])) * 60, 2)
        self.__eyeXL = self.__eyeXR
        self.__eyeYL = self.__eyeYR 

        # adjust eyes X location when close to head
        # prominence is the amount of degree you 
        # rotate the eyes towards each other
        prominence = 10 - ((z/1000)*10)
        if prominence < 0:
            prominence = 0
        if prominence > 10:
            prominence = 10
        self.__eyeXR -= prominence
        self.__eyeXL += prominence

    def __calc_2D_simple(self,x,y,z):
        self.__eyeXR = round(-30 + (x / self.__camera_resolution[0]) * 60, 2)
        self.__eyeYR = round(-30 + (1-(y / self.__camera_resolution[1])) * 60, 2)
        self.__eyeXL = self.__eyeXR
        self.__eyeYL = self.__eyeYR            
    
    def get_eye_coordinates(self):
        # returns a string:'eye_right_x,eye_right_y,e_left_x,e_left_y' 
        #  with coordinates for eyes            
        return f'{self.__eyeXR},{self.__eyeYR},{self.__eyeXL},{self.__eyeYL}'

    def set_xyz(self, x , y , z = 1000):
        self.__calc_eye_coordinates(x,y,z)

    def set_mode(self,mode_str):
        self.__mode = mode_str

    def set_eye_center_offset_from_screen(self, distance_z):
        self.__eye_center_offset_from_screen = distance_z

    def set_camera_resolution(self,resolution_tuple):
        # takes in a tuple in this format (x,y) , (640,480)
        self.__camera_resolution = resolution_tuple


def main():
    print('test')

if __name__ == '__main__':
    main()
