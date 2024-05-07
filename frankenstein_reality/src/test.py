#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class LaserScanPlotter:
    def __init__(self):
        rospy.init_node('laser_scan_plotter', anonymous=True)
        self.sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.angles = [0, np.pi/2, np.pi, -np.pi/2]
        self.range_data = {angle: [] for angle in self.angles}
        self.intensity_data = {angle: [] for angle in self.angles}
        self.fig, self.axs = plt.subplots(2, 2)
        self.plots = {}
        self.setup_plots()

    def setup_plots(self):
        for i, angle in enumerate(self.angles):
            ax = self.axs[i//2, i%2]
            self.plots[angle] = {
                'range': ax.plot([], [], label='Range')[0],
                'intensity': ax.plot([], [], label='Intensity')[0]
            }
            ax.set_title(f'Angle: {np.degrees(angle)} degrees')
            ax.legend()

    def scan_callback(self, msg):
        current_angle_index = np.array([(angle - msg.angle_min) / msg.angle_increment for angle in self.angles])
        for angle, idx in zip(self.angles, current_angle_index):
            idx = int(idx)
            if 0 <= idx < len(msg.ranges):
                self.range_data[angle].append(msg.ranges[idx])
                self.intensity_data[angle].append(msg.intensities[idx] if msg.intensities else 0)

    def update_plot(self, frame):
        for angle in self.angles:
            range_plot = self.plots[angle]['range']
            intensity_plot = self.plots[angle]['intensity']
            x_data = np.arange(len(self.range_data[angle]))
            range_plot.set_data(x_data, self.range_data[angle])
            intensity_plot.set_data(x_data, self.intensity_data[angle])
            ax = range_plot.axes  # Get the axes for this plot

            # Update the limits
            ax.set_xlim(0, max(5000, len(self.range_data[angle]) + 10))  # Adjust as more data comes in
            all_values = self.range_data[angle] + self.intensity_data[angle]
            if all_values:  # Prevent error if all_values is empty
                ax.set_ylim(0, max(all_values) * 1.1)  # Scale y-axis to 10% above max value seen

        self.fig.canvas.draw()
        return self.plots

    def start(self):
        ani = FuncAnimation(self.fig, self.update_plot, interval=100)
        plt.show()

if __name__ == '__main__':
    plotter = LaserScanPlotter()
    try:
        plotter.start()
    except rospy.ROSInterruptException:
        pass
    finally:
        plt.close(plotter.fig)
