import math
import numpy as np


class ConvertCoordinates():
    """
    Class that recieves pixel X Y and depth from camera. Camera to eye offsets.
    converts detected focus point to eye simulator eye X and Y position
    for left and right eye individually.

    Will have different options of complexity of data handling

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
        self.__camera_to_between_eyes_offset_x = 0  
        self.__camera_to_between_eyes_offset_y = 120 
        self.__camera_to_between_eyes_offset_z = 0  
        self.__eye_offset_R_x = 31  
        self.__eye_offset_L_x = -31  
        self.__eye_offset_R_y = self.__camera_to_between_eyes_offset_y 
        self.__eye_offset_L_y = self.__camera_to_between_eyes_offset_y
        self.__x_range_eyes = (30, -30)  # left to right looking at the eyes
        self.__y_range_eyes = (-30, 30)  # down to up
        self.__eye_center_offset_from_screen = -10 #
        self.__camera_x = None
        self.__camera_y = None
        self.__camera_resolution = (640,480)
        self.__mm_per_pixel_x = 10
        self.__mm_per_pixel_y = 10 #function of depth to plane.
        self.__fov_x = 64 #field of view[degree] horisontal axis ( x)
        self.__fov_y = 41  #field of view[degree] of view vertical axis ( y)


    def __calc_eye_coordinates(self,x,y,z):
        if self.__mode == '3D':
            self.__calc_3D(x,y,z)
        elif self.__mode == '2D':
            self.__calc_2D(x,y,z)
        else:
            self.__calc_2D_simple(x,y,z)

    def __calc_3D(self,x,y,depth):
        #TODO
        """
        X pixel for detection object to focus on
        Y pixel for detection object to focus on
        depth mm to detected object to focus on

        takes in pixel coordinates and depth and sets eye X,Y rotation (degrees from 0,0 straight ahead)
        """

        #Calculate X,Y angle relative to eye references
        pix_per_degree_x = self.__camera_resolution[0]/self.__fov_x #640/64= 10
        degrees_from_left = x / pix_per_degree_x   # 120 / 10 = 12 degree
        degrees_from_center = degrees_from_left - (self.__fov_x/2) #12 - 32 = -20 degree
        x_distance_from_center_mm = math.sin(degrees_from_center) * depth #  340mm

        pix_per_degree_y = self.__camera_resolution[1]/self.__fov_y #480/41= 11.70
        degrees_from_top = y / pix_per_degree_y   # 240 / 11.7 = 20.51
        degrees_from_center = degrees_from_top - (self.__fov_y/2) #20.51 - 20.5 = 0.01 degree
        y_distance_from_center_mm = math.sin(degrees_from_center) * depth

        z_distance_from_center_mm = math.cos(degrees_from_center)* depth

        #x and y coordinates relative to eye positions
        left_eye_x = x_distance_from_center_mm - self.__eye_offset_L_x 
        left_eye_y = y_distance_from_center_mm - self.__eye_offset_L_y 

        right_eye_x = x_distance_from_center_mm - self.__eye_offset_R_x 
        right_eye_y = y_distance_from_center_mm - self.__eye_offset_R_y 

        left_eye_x_angle = math.asin(left_eye_x/z_distance_from_center_mm)
        left_eye_y_angle = math.asin(left_eye_y/z_distance_from_center_mm)

        right_eye_x_angle = math.asin(right_eye_x/z_distance_from_center_mm)
        right_eye_y_angle = math.asin(right_eye_y/z_distance_from_center_mm)


        #set eye angles for adafruit eyes.
        self.__eyeXR = right_eye_x_angle
        self.__eyeYR = right_eye_y_angle
        self.__eyeXL = left_eye_x_angle
        self.__eyeYL = left_eye_y_angle       

        



    def __calc_3D_xyz_intersect(self, x,y,z):
        #TODO

        #Calculate cartesian X location camera reference
        pix_per_degree_x = self.__camera_resolution[0]/self.__fov_x #640/64= 10
        degrees_from_left = x / pix_per_degree_x   # 120 / 10 = 12 degree
        degrees_from_center = degrees_from_left - (self.__fov_x/2) #12 - 32 = -20 degree
        x_distance_from_center_mm = math.sin(degrees_from_center) * z #  340mm
        

        #calculate cartesian Y location camera reference
        pix_per_degree_y = self.__camera_resolution[1]/self.__fov_y #480/41= 11.70
        degrees_from_top = y / pix_per_degree_y   # 240 / 11.7 = 20.51
        degrees_from_center = degrees_from_top - (self.__fov_y/2) #20.51 - 20.5 = 0.01 degree
        y_distance_from_center_mm = math.sin(degrees_from_center) * z #  9.99

        #xyz = (340,-9.99,1000) from camera
        focus_spot_position = np.array[(x_distance_from_center_mm,y_distance_from_center_mm,z)] # position coordinate
        #calculate Eye vectors. Eye position = left eye LE = (-31,120,-10)  right eye = (31,120,-10)
        left_eye_position = np.array[(self.__eye_offset_L_x,self.__camera_to_between_eyes_offset_y,self.__camera_to_between_eyes_offset_z)]
        right_eye_position = np.array[(self.__eye_offset_R_x,self.__camera_to_between_eyes_offset_y,self.__camera_to_between_eyes_offset_z)]

        left_eye_vector = focus_spot_position-left_eye_position
        right_eye_vector = focus_spot_position-right_eye_position


        #calculate vector and eye plane intersection.
        #intersection happens when z is 0 of eye vectors. will need to parameterize the eye vectors. 
        # Get component forms for x and y where z = 0.
        # parameterized fuctions:
        # example: focus_spot_position + t * focus_spot_position  
        #find t where Z = 0 and put it into the other  dimensions:

        # t where z is zero - focus spot position negative z vector divided by left eye vectors z displacement
        left_eye_t_zero_z = (-focus_spot_position[2]) / left_eye_vector[2]

        left_eye_x_zero_z = focus_spot_position[0] + ( left_eye_vector[0] * left_eye_t_zero_z )
        left_eye_y_zero_z = focus_spot_position[1] + ( left_eye_vector[1] * left_eye_t_zero_z )

        #same for right eye 
        right_eye_t_zero_z = (-focus_spot_position[2]) / right_eye_vector[2]

        right_eye_x_zero_z = focus_spot_position[0] + ( right_eye_vector[0] * right_eye_t_zero_z )
        right_eye_y_zero_z = focus_spot_position[1] + ( right_eye_vector[1] * right_eye_t_zero_z )

        #find pixel positions for eye screens. pixel 0, 0 is in location relative to 



        # self.__eyeXR = 
        # self.__eyeYR = 
        # self.__eyeXL = 
        # self.__eyeYL = 

        
    def __calc_2D(self, x,y,z):
        self.__eyeXR = round(-30 + (x / self.__camera_resolution[0]) * 60, 2)
        self.__eyeYR = round(-30 + (1-(y / self.__camera_resolution[1])) * 60, 2)
        self.__eyeXL = self.__eyeXR
        self.__eyeYL = self.__eyeYR 

        #adjust eyes X location when close to head
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
