import json
import os
import bisect
import dlib
import cv2
import copy
import traceback
import numpy as np
import pandas as pd
import unicodecsv as ucsv
import matplotlib.pyplot as plt
from keras.engine.saving import model_from_json
from keras_preprocessing.image import img_to_array

# Replacement of SimpleNameSpace, which is only available in Python 3
class Namespace (object):
    def __init__ (self, **kwargs):
        self.__dict__.update(kwargs)
    def __repr__ (self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))
    def __eq__ (self, other):
        return self.__dict__ == other.__dict__

# Function calculating euclidean distance of two points
def dist(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

# Function comparing the total distance of all landmarks between two faces
# Useful when recognizing if a face detected is the intended person
def euclidean(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

# Function converting a rectangular image to a square one whie preserving its ratio
# Does so by adding black rectangles on the sides of the longer edge of the rectangle
# example: 400*200 image input -> 400*400 image output with 400*100 black rectangles at the top and the bottom
def square_img(img):
    s = max(img.shape[:2])
    f = np.zeros((s, s, 3), np.uint8)
    x, y = (s - img.shape[0]) // 2, (s - img.shape[1]) // 2
    f[x:img.shape[0] + x, y:img.shape[1] + y] = img
    return f

# Function to analyze a video
def analyze_video(raw_name, target, writer=None, out_dir=None):
    name = raw_name + ".mov"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    else:
        os.system("rm -f " + out_dir + "/*")

    # reading model data
    vid = cv2.VideoCapture(name)
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Parsing video " + name)
    predictor_path = r'files'
    cascade = cv2.CascadeClassifier(predictor_path + r"/haarcascade_frontalface_default.xml")
    predictor = dlib.shape_predictor(predictor_path + r"/shape_predictor_68_face_landmarks.dat")
    model = model_from_json(
        open(r"files/facenet_model.json", "r").read())

    # reading and analyzing image of the face we want to detect
    target_img = cv2.imread(target, cv2.IMREAD_GRAYSCALE)
    target_face = cascade.detectMultiScale(target_img)
    target_detected = None
    for (x, y, w, h) in target_face:
        target_detected = cv2.resize(target_img[int(y):int(y + h), int(x):int(x + w)], (160, 160))
    target_detected = cv2.cvtColor(target_detected, cv2.COLOR_GRAY2BGR)
    target_restored = np.expand_dims(target_detected, axis=0)
    target_prediction = model.predict(target_restored)

    # read in video and analyze frame by frame
    duration, detection = [], []
    num = 0
    fps = vid.get(cv2.CAP_PROP_FPS)
    while True:
        success, img = vid.read()
        if not success:
            break
        elif num >= 300:
            break
        face = cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.3, 5)
        timestamp = num / fps

        # find all faces of appropriate size and compare them to the target face
        msc = 4096 * 4096
        detected = None
        detected_cdn = None
        for (x, y, w, h) in face:
            if 200 < w < 700:
                temp_detected = cv2.resize(square_img(img[int(y):int(y + h), int(x):int(x + w)]), (160, 160))
                temp_captured = model.predict(np.expand_dims(temp_detected, axis=0))
                distance = euclidean(temp_captured, target_prediction)

                if distance < msc:
                    msc = distance
                    detected_face = temp_detected
                    detected_cdn = (x, y, w, h)

        # if no desired face can be found, mark the frame as undetected/error
        detected = -1
        if detected is None or msc > 60:
            detection.append(0)
            detected = 0
            duration.append(-0.01)
            text = "Detection: 0"
            cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.imwrite(out_dir + "/" + str(num) + ".jpg", img)
        else: # if found, proceed with analyzing lip landmarks
            detection.append(1)
            detected = 1
            shape = predictor(detected_face, dlib.rectangle(int(0), int(0), int(160), int(160)))

            i = 0
            (x, y, w, h) = detected_cdn
            left, right, top, bottom = 0, 0, 0, 0
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            for pt in shape.parts():
                i += 1
                if i >= 49:
                    cv2.circle(img, (pt.x + x, pt.y + y), 2, (0, 255, 0), 2)
                    if i == 61:
                        left = pt
                    elif i == 65:
                        right = pt
                    elif i == 63:
                        top = pt
                    elif i == 67:
                        bottom = pt

            # analyze the ratio: ((bottom of top lip - top of bottom lip) / (right edge of mouth - left edge of mouth))
            rate = dist(top, bottom) / dist(left, right)
            duration.append(rate)
            text = "Detection: 1; Distance: " + str(rate)
            if rate >= 0.05:
                cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            else:
                cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            # uncomment the following line to output frame images of a demo video
            # cv2.imwrite(out_dir + "/" + str(num) + ".jpg", cv2.resize(img, (640, 480)))

        # write frame data into .csv file
        if detected == 0:
            detected_cdn = (-1, -1, -1, -1)
            left = type("", (), dict(x=-1, y=-1))
            top = right = bottom = left
            rate = -0.01
        if writer is not None:
            writer.writerow(
                [num, timestamp, detected, detected_cdn[0], detected_cdn[1], detected_cdn[2], detected_cdn[3],
                 left.x, left.y, top.x, top.y, right.x, right.y, bottom.x, bottom.y, rate])
        num += 1
    return duration, detection, length

# calculate acceleration of lip movement (unimportant as of now)
def calcSpeed(duration, detection):
    acc = []
    for i in range(0, len(duration)):
        front, back = 0, 0
        if i == len(duration) - 1:
            front = duration[i] + (duration[i] - duration[i - 1])
        else:
            front = duration[i + 1]
        if i == 0:
            back = duration[i] - (duration[i] - duration[i - 1])
        else:
            back = duration[i - 1]
        acc.append(front - back)
    return acc

# wrapper of analyze_video
def process_video(filepath, filename, profile_name):
    # filepath = "E:\\Participant #1\\videos\\Week 2\\Day 5\\" # "database\\"#
    args = ["Frame", "Timestamp", "Detection", "Face_x", "Face_y", "Face_w", "Face_h", "Mark_61_x", "Mark_61_y",
            "Mark_63_x", "Mark_63_y", "Mark_65_x", "Mark_65_y", "Mark_67_x", "Mark_67_y", "Distance"]
    proc_args = ["Slot_num", "State"]

    profile = profile_name + ".jpg"
    filedir = filepath + filename
    length = 0
    duration = []
    timestamps = []
    wr = None

    # check if all relevant directories exists and are empty
    if not os.path.exists(profile_name + "_Lip_Data/"):
        os.mkdir(profile_name + "_Lip_Data/")
    if not os.path.exists(profile_name + "_Lip_Data/" + filename + "/"):
        os.mkdir(profile_name + "_Lip_Data/" + filename + "/")
    else:
        os.system("rm -f " + profile_name + "_Lip_Data/" + filename + "/*")

    # if no .csv files can be found, analyze the videos
    if not os.path.exists(profile_name + "_Lip_Data/" + filename + "/" + filename + "_Video_Processed.csv"):
        if not os.path.exists(profile_name + "_Lip_Data/" + filename + "/" + filename + "_Video_Raw.csv"):
            with open(profile_name + "_Lip_Data/" + filename + "_Video_Raw.csv", "wb") as f:
                wr = ucsv.writer(f)
                wr.writerow(args)
                duration, detection, length = analyze_video(filedir, profile, writer=wr,
                                                out_dir=(profile_name + "_Lip_Data/" + filename + "/"))

        with open(profile_name + "_Lip_Data/" + filename + "/" + filename + "_Video_Raw.csv", "r") as f:
            # else if only raw .csv files can be found, read raw data from them and analyze them
            category = f.readline().split(",")
            lines = f.readlines()
            curr = 0.0
            timeslot = []
            state = 0
            for line in lines:
                content = line.split(",")
                timestamp = (float(content[1])) // 0.2
                timestamps.append(float(content[1]))
                if curr > timestamp:
                    continue
                elif curr + 1 <= timestamp:
                    curr += 1
                    timeslot.append(state)
                    state = 0
                if int(content[2]) == 1:
                    rate = float(content[15])
                    if wr is None:
                        duration.append(rate)
                    if rate >= 0.05:
                        state = 2
                elif state == 0:
                    state = 1
                    if wr is None:
                        duration.append(-0.01)
        # write the analyzed results into processed .csv files
        with open(profile_name + "_Lip_Data/" + filename + "/" + filename + "_Video_Processed.csv", "wb") as f:
            wr = ucsv.writer(f)
            wr.writerow(proc_args)
            for i in range(0, len(timeslot)):
                wr.writerow([i, timeslot[i]])
    else:
        # if analyzed .csv files can be found, use them directly
        with open(profile_name + "_Lip_Data/" + filename + "/" + filename + "_Video_Raw.csv", "r") as f:
            category = f.readline().split(",")
            lines = f.readlines()
            curr = 0.0
            timeslot = []
            state = 0
            for line in lines:
                content = line.split(",")
                t = float(content[1])
                timestamps.append(t)
                duration.append(float(content[15]))

    slots = []

    # if analyzed .csv files can be found, use them directly
    with open(profile_name + "_Lip_Data/" + filename + "/" + filename + "_Video_Processed.csv", "r") as f:
        category = f.readline().split(",")
        lines = f.readlines()
        for line in lines:
            content = line.split(",")
            slots.append(int(content[1]))

    # draw plots of lip movement and detection results, with each plot featuring one minute of video
    num = max(1, (len(slots) + 299) // 300)
    plt.rcParams.update({"font.size": 22})
    for i in range(0, num):
        fig = plt.figure(num=None, figsize=(20, 8), dpi=100)
        axs = fig.add_subplot(111)
        curr = i * 300
        next = len(slots) if i == num - 1 else (i + 1) * 300
        tc = bisect.bisect_left(timestamps, curr * 0.2)
        tn = bisect.bisect_left(timestamps, next * 0.2)
        axp = axs.twinx()
        axs.set_xlabel("Time (s)")
        axp.set_ylabel("Distance", color="#0000ff")
        axp.set_ylim(-0.025, 0.425)
        axs.set_ylabel("State of timeslot", color="#ff9f00")
        axs.set_ylim(-0.133, 2.267)
        axs.step(np.arange(0.2 * curr, 0.2 * next, 0.2), slots[curr:next], linewidth=1, where="post", label="Filtered",
                 color="#ff9f00", zorder=-10)
        axp.plot(timestamps[tc:tn], duration[tc:tn], "-o", linewidth=1, markersize=1.5, label="Raw data",
                 color="#0000ff", zorder=10)
        axp.plot([curr * 0.2, next * 0.2], [0.05, 0.05], "--", linewidth=2.5, color="#007f00", zorder=10)
        axs.fill_between(np.arange(0.2 * curr, 0.2 * next, 0.2), slots[curr:next], 0, color="#ffcf9f", zorder=-10)
        plt.savefig(filepath + "Plots/Fig_" + str(i) + ".png", bbox_inches="tight")


'''

'''

def main():
    # iterate through directories mentioned in this json file to analyze all videos necessary
    filename = "/home/team/NRI-Kids_videos-tree.json"
    with open(filename, "r") as f:
        line = f.readlines()[0]
        all_files = json.loads(line, object_hook=lambda d: Namespace(**d))
    files = all_files.__dict__["Participant_2"].videos

    for week in files.__dict__:
        try:
            for name in files.__dict__[week].__dict__["."]:
                # print(week + "/" + name)
                pathname = "Participant_2/" + week + "/"
                print ("Analyzing " + pathname + name)
                process_video(pathname, name, "Participant_2")
        except Exception, e:
            print ("Participant_2/" + week + " cannot be found")
            print(str(e))
            traceback.print_exc()
        for day in files.__dict__[week].__dict__.keys():
            try:
                for name in files.__dict__[week].__dict__[day].__dict__["."]:
                    pathname = "Participant_2/" + week + "/" + day + "/"
                    print ("Analyzing " + pathname + name)
                    process_video(pathname, name, "Participant_2")
            except Exception, e:
                print ("Participant_2/" + week + "/" + day + " cannot be found")
                print(str(e))
                traceback.print_exc()

if __name__ == "__main__":
    main()