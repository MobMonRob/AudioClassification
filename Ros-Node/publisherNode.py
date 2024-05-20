#!/usr/bin/env python

import rospy
import librosa
import numpy
from std_msgs.msg import String

from calculateRPM import calculateRPMS
from calculateGrindingStatus import calculateGrindingStatus



def calculateData(data, sample_rate):
    rpmsMessuredForEachSecond, intensitysOfRPM = calculateRPMS(data, sample_rate)

    return rpmsMessuredForEachSecond, intensitysOfRPM

def read_data_from_source(PATH, sr):

    sst, sample_rate = librosa.load(PATH, sr=sr)

    return sst, sample_rate

def grinding_audio_data_publisher():
    rospy.init_node('grinding_audio_data_publisher', anonymous=True)
    
    pub = rospy.Publisher('grinding_data', String, queue_size=10)
    
    # Frequenz, mit der Nachrichten veröffentlicht werden sollen in Hz
    rate = rospy.Rate(10)

    PATH = "ChangeThisToYourPath"

    if PATH == "ChangeThisToYourPath":
        rospy.loginfo(f"PATH to data is not defined in the node")

    data, sample_rate = read_data_from_source(PATH, 8000)
    rpmsMessuredForEachSecond, intensitysOfRPM = calculateData(data, sample_rate=sample_rate)
    grindingStatus = calculateGrindingStatus(data, sr=sample_rate)

    seconds = numpy.min(len(rpmsMessuredForEachSecond), len(grindingStatus))

    rospy.loginfo(f"Publishing grinding data: ")

    for i in range(0, seconds-1):
        pub.publish("Second: "+ i + "RPM: " + rpmsMessuredForEachSecond[i] + ", AppearanceIntensity: " + intensitysOfRPM[i] + "GrindingStatus: " + grindingStatus[i])
        
        # Schlafe, um die Frequenz einzuhalten
        rate.sleep()

if __name__ == '__main__':
    try:
        grinding_audio_data_publisher()
    except rospy.ROSInterruptException:
        pass
