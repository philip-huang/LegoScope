#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from find_cam import find_cam

def main():
    # Get camera
    camera_ids = find_cam()
    if not camera_ids:
        rospy.logerr("No camera detected.")
        return
    camera = cv2.VideoCapture(camera_ids[-1])
    if not camera.isOpened():
        rospy.logerr("Failed to open camera.")
        return

    # ROS setup
    rospy.init_node('camera_compressed_publisher', anonymous=True)
    pub = rospy.Publisher('/camera/image/compressed', CompressedImage, queue_size=1)
    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        ret, frame = camera.read()
        if not ret:
            rospy.logwarn("Failed to grab frame from camera.")
            continue

        # Encode frame as JPEG
        success, encoded_img = cv2.imencode('.jpg', frame)
        if not success:
            rospy.logwarn("Failed to encode image")
            continue

        # Create CompressedImage message
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = encoded_img.tobytes()

        pub.publish(msg)
        rate.sleep()

    camera.release()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
