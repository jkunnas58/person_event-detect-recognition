import socket

class RecieveData():
    """ 
    Class that starts a socket connection and recieves eye coordinates
    for eye simulator to use
    
    """
    def __init__(self):
        self.__host = '192.168.191.125'
        self.__port = 65432
        self.__eyeXR = 30    
        self.__eyeYR = 30
        self.__eyeXL = 30    
        self.__eyeYL = 30
        self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)   
        self.__connected_to_socket = False
           
    def process_data_from_server(self,x):
	    self.__eyeXR , self.__eyeYR , self.__eyeXL , self.__eyeYL = x.split(",")
        

    def connect_to_server(self):     
        try:
            self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)       
            self.__socket.connect((self.__host, self.__port))  
            self.__connected_to_socket = True
        except:
            self.__connected_to_socket = False

    def get_data_from_connection(self):
            data = self.__socket.recv(1024).decode('utf-8')
            self.process_data_from_server(data)


    def get_eye_coordinates_str(self):
	    self.get_data_from_connection()
	    return  self.__eyeXR,\
                self.__eyeYR,\
                self.__eyeXL,\
                self.__eyeYL 

    def get_eye_coordinates_float(self):
	    self.get_data_from_connection()
	    return  float(self.__eyeXR),\
                float(self.__eyeYR),\
                float(self.__eyeXL),\
                float(self.__eyeYL)

    def close_socket(self):
        self.__socket.shutdown()
        self.__socket.close()

    def get_socket_connected_status(self):
        return self.__connected_to_socket

    def set_socket_connected_status(self, bool):
        self.__connected_to_socket = bool

    def set_host_ip(self, host_ip):
        # Set host ip as string: example: '192.168.2.1'
        self.__host = host_ip

    def set_host_port(self, host_port):
        # Set host port as integer: example: 65432
        self.__port = host_port

def main():
    eye_coordinates = RecieveData()
    eye_coordinates.connect_to_server()
    while True:
        eyex,eyey,eyex2,eyey2 = eye_coordinates.get_eye_coordinates_str()
        print(f'EyeX: {eyex} , EyeY: {eyey}, EyeX2: {eyex2} , EyeY2: {eyey2}')

if __name__ == "__main__":
    while True:
        main()