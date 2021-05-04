import socket
import numpy as np
import time


class SendData():
    """
    Send data class. This class sets up a sending server waiting for clients to
    connect.

    class has a class method that sets up the server in setup_server_sending()

    The function send_data(data_to_send) sends the data as string in utf-8
    encoding 

    """

    def __init__(self) -> None:
        self.__host = '192.168.191.119' # loopback interface address (localhost)
        self.__port = 65432 # Port to listen on (non-privileged ports are > 1023)
        self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__connection = None
        self.__address = None

    def setup_server_sending(self):
        print("Server Started waiting for client to connect ")
        self.__socket.bind((self.__host, self.__port))
        self.__socket.listen(5)
        self.__connection, self.__address = self.__socket.accept()
        print('Connected to', self.__address)

    def send_data(self,my_data):
        # my_data = f'{self.__eyeX},{self.__eyeY}'
        # print(my_data)
        my_data_bytes = bytes(my_data, 'utf-8')
        # print('length of bytes: ', len(my_data_bytes))
        self.__connection.send(my_data_bytes)

    def set_host_ip(self, ip):
        #set host ip as string '192.168.1.1'
        self.__host = ip

    def set_port(self, port):
        #set port as int
        self.__port


class RandomData():

    def __init__(self) -> None:
        self.oldtime = time.time()
        self.x1 = np.random.randint(-30, 30, None)
        self.y1 = np.random.randint(-30, 30, None)

    def random_data(self):
        if time.time() - self.oldtime > 2:
            x1 = np.random.randint(-30, 30, None)         # Dummy eye x
            y1 = np.random.randint(-30, 30, None)        # Dummy dummy eye y
        else:
            x1 = self.x1
            y1 = self.y1

        return x1, y1

def main():
    random_data = RandomData()
    send_data = SendData()    
    send_data.setup_server_sending()

    while True:
        # eye_x = 0   # 30 left to -30 right looking at the dool
        # eye_y = 0 # -30 looking down 30 looking up
        eye_x, eye_y = random_data.random_data()
        send_data.send_data(f'{eye_x},{eye_y},{eye_x},{eye_y}')
        time.sleep(0.5)


if __name__ == '__main__':
    main()

