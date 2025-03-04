#!/usr/bin/env python

import rospy
from geometry_msgs.msg import WrenchStamped

def callback(msg):
    force = msg.wrench.force
    torque = msg.wrench.torque
    rospy.loginfo("Force: x=%.2f, y=%.2f, z=%.2f | Torque: x=%.2f, y=%.2f, z=%.2f",
                  force.x, force.y, force.z, torque.x, torque.y, torque.z)
    print("hi")

def listener():
    rospy.init_node('fts_subscriber', anonymous=True)
    rospy.Subscriber("/yk_destroyer/fts", WrenchStamped, callback)
    rospy.spin()

if __name__ == '__main__':
    print("18")
    listener()
