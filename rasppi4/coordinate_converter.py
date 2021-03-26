



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


    def __calc_eye_coordinates(self,x,y,z):
        if self.__mode == '3D':
            self.__calc_3D(x,y,z)
        elif self.__mode == '2D':
            self.__calc_2D(x,y,z)
        else:
            self.__calc_2D_simple(x,y,z)

    def __calc_3D(self, x,y,z):
        #TODO
        self.__eyeXR = round(-30 + (x / self.__camera_resolution[0]) * 60, 2)
        self.__eyeYR = round(-30 + (1-(y / self.__camera_resolution[1])) * 60, 2)
        self.__eyeXL = self.__eyeXR
        self.__eyeYL = self.__eyeYR

        
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
