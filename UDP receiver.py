import socket

ip = ('127.0.0.1',5000)

sudp = socket.socket(socket.AF_INET,socket.SOCK_DGRAM,0)

sudp.bind(ip)

while True:

    bdata = sudp.recv(1024)

    print (bdata.decode('utf-8'))