import socket
import struct
import rospy
from sensor_msgs.msg import LaserScan

file = '/home/aminedhemaied/catkin_ws/src/frankenstein/frankenstein_reality/data/lidar_data3.txt' #fitxer on guardem les dades

#primer he de moure l'AGV fins que la localizacio quedi ben agafada!!!!
# - fer una ruta curta assegurant-me del pas anterior
# - fer una ruta quadrat per veure que tal va
# - fer una ruta lliure llarga per tal de poder comprobar amb girs en dos sentits.

def write_data_udp():
    HOST = '172.16.1.253'
    PORT = 5009

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind((HOST, PORT))
        data = s.recvfrom(1024)
        values = struct.unpack('@Qdddddddddddd', data[0])

        line_to_write = f"{values[0]},{values[1]},{values[2]},{values[3]},{values[7]},{values[8]},{values[9]}\n"
        print(f"{values[0]},{values[1]},{values[2]},{values[3]},{values[7]},{values[8]},{values[9]}\n")
        print("Storing Odom")
        f = open(file,"a")
        f.write(line_to_write)
        f.close()

def laser_scan_callback(msg):
    ranges = msg.ranges
    angles = []
    angle_min = msg.angle_min
    angle_increment = msg.angle_increment

    write_data_udp()
    f = open(file,"a")
    print("Storing Lidar")
    lidar_data = []
    for i, distance in enumerate(ranges):
        angle = angle_min + i * angle_increment
        angles.append(angle)
        lidar_data.append(f"{distance},{angle},")

    for data in lidar_data:
        f.write(data)
    f.write('\n')

    f.close()

def laser_scan_subscriber():
    rospy.init_node('laser_scan_subscriber', anonymous=True)
    rospy.Subscriber('/scan', LaserScan, laser_scan_callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        laser_scan_subscriber()
    except rospy.ROSInterruptException:
        pass