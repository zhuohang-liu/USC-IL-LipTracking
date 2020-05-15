import os
import dlib
import cv2
import copy
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from keras.engine.saving import model_from_json
from keras_preprocessing.image import img_to_array

def dist(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

def normalize(im):
    im = img_to_array(im)
    im /= 127.5
    im -= 1
    return im

def euclidean(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def findInstance(duration, offset=0, bias: float = 0.5, base: int = 0):
    inst = []
    started = -1
    i = 0
    while i < len(duration):
        if abs(duration[i]) >= bias and started < 0:
            started = i
            i += 1
        elif abs(duration[i]) < bias and started >= 0:
            ind = i
            while i < min(len(duration), ind + 15) and abs(duration[i]) < bias:
                i += 1
            if i == min(len(duration), ind + 15):
                inst.append([started + offset, ind + offset, 2])
                started = -1
                i += 1
        else:
            i += 1
    if started is not -1 and len(duration) - started >= 30:
        inst.append([started + offset, len(duration) + offset, 2])
    inst = fill(inst, len(duration), bias=offset, base=base)
    return inst


def rolling_mean(duration, windows, length: int):
    temp = pd.DataFrame(copy.deepcopy(duration))
    w1 = temp # temp.rolling(window=1).mean()
    w2 = temp.rolling(window=10).mean()
    duration_w1 = [abs(i[0]) for i in w1.values.tolist()]
    duration_w2 = [abs(i[0]) for i in w2.values.tolist()[10:]]
    for _ in range(0, 10):
        duration_w2.insert(0, duration_w2[0])

    # xl = [*range(0, length)]
    # fig = plt.figure(num=None, figsize=(20, 8), dpi=100)
    # ax = fig.add_subplot(111)
    # ax.plot(xl, duration_w1, "-o", linewidth=0.5, markersize=0.8)
    # ax.plot(xl, duration_w2, linewidth=0.5)
    # plt.show()

    fig = plt.figure(num=None, figsize=(20, 8), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot([*range(0, length)], [w1-w2 for (w1, w2) in zip(duration_w1, duration_w2)], "--", linewidth=0.2)
    ax.plot([*range(0, length)], [0.01 for _ in range(0, length)], "--", linewidth=0.2)
    plt.show()

    return duration_w1, duration_w2


def findAbove(duration, length: int, offset=0, bias: float=0.05, window: int=10, base: int=0):
    inst = []
    started = -1
    duration_w1, duration_w2 = rolling_mean(duration, length=length, windows=window)
    i = 0
    while i < len(duration):
        if abs(duration_w1[i]) >= abs(duration_w2[i] + bias) and started < 9:
            started = i
            i += 1
        elif abs(duration_w1[i]) < abs(duration_w2[i]) and started >= 0:
            ind = i
            while i < min(len(duration), ind + 30):
                i += 1
            if i == min(len(duration), ind + 30):
                inst.append([started + offset, ind + offset, 2])
                started = -1
                i += 1
        else:
            i += 1
    if started is not -1 and len(duration) - started >= 15:
        inst.append([started + offset, len(duration) + offset, 2])
    inst = fill(inst, len(duration), bias=offset, base=base)
    return inst


def inspectInstance(inst, acc, base_acc=0.25):
    replaced = {}
    for ii in inst:
        if ii[2] == 2:
            piece = acc[ii[0]:ii[1]]
            temp = findInstance(piece, offset=ii[0], bias=base_acc, base=1)
            temp = fill(temp, len(piece), bias=ii[0], base=1)
            replaced[tuple(ii)] = temp
    for (k, v) in zip(replaced.keys(), replaced.values()):
        ind = inst.index(list(k))
        inst.remove(list(k))
        for i in range(len(v) - 1, -1, -1):
            inst.insert(ind, v[i])
    return inst


def fill(inst, length, bias: int = 0, base: int = 0):
    curr = bias
    for ii in inst:
        if curr != ii[0]:
            inst.insert(inst.index(ii), [curr, ii[0], base])
            curr = ii[0]
        else:
            curr = ii[1]
    inst.append([curr, length + bias, base])
    return inst


def process(duration, acc, min_dist: float = 0.5, min_speed: float = 0.25):
    inst = findInstance(duration, bias=min_dist)
    inst = inspectInstance(inst, acc, base_acc=min_speed)
    inst = fill(inst, len(duration))
    for i in range(len(inst)-1, 0, -1):
        if inst[i][0] == inst[i][1]:
            inst.pop(i)
    return inst


def process_alt(duration, acc, length: int, bias: float=0.5, min_speed: float=0.25,):
    inst = findAbove(duration, length=length, bias=bias)
    inst = inspectInstance(inst, acc, base_acc=min_speed)
    inst = fill(inst, len(duration))
    for i in range(len(inst)-1, 0, -1):
        if inst[i][0] == inst[i][1]:
            inst.pop(i)
    return inst

def square_img(img):
    s = max(img.shape[:2])
    f = np.zeros((s, s, 3), np.uint8)
    x, y = (s-img.shape[0]) // 2, (s-img.shape[1]) // 2
    f[x:img.shape[0]+x, y:img.shape[1]+y] = img
    return f


def analyze_video(raw_name, target):

    name = raw_name + ".mov"
    # out_name = raw_name + "_Output.avi"
    out_dir = "D:/Participants/Out_Imgs"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    else:
        os.system("rm -f " + out_dir + "/*")

    vid = cv2.VideoCapture(name)
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Parsing video " + name)
    print("Building detection tools...", end='')
    predictor_path = r'C:\Users\liu_z\PycharmProjects\Facenet-realtime\files'
    cascade = cv2.CascadeClassifier(predictor_path + r"\haarcascade_frontalface_default.xml")
    # base_dir = os.path.dirname(__file__)
    # protopath = os.path.join(base_dir, r"files\deploy.prototxt")
    # caffepath = os.path.join(base_dir, r"files\weights_alt.caffemodel")
    # dnn_model = cv2.dnn.readNetFromCaffe(protopath, caffepath)
    predictor = dlib.shape_predictor(predictor_path + r"\shape_predictor_68_face_landmarks.dat")
    print("done")

    print("Building face prediction model...", end='')
    model = model_from_json(
        open(r"C:\Users\liu_z\PycharmProjects\Facenet-realtime\files/facenet_model.json", "r").read())
    print("done")

    # writer = cv2.VideoWriter(out_name,
    #     cv2.VideoWriter_fourcc(*"MJPG"), 30, (720, 1080))

    target_img = cv2.imread(target, cv2.IMREAD_GRAYSCALE)
    # target_img = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR)
    # target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    target_face = cascade.detectMultiScale(target_img)
    # cv2.imwrite("gray.jpg", target_img)
    # target_blob = cv2.dnn.blobFromImage(cv2.resize(target_img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    # dnn_model.setInput(target_blob)
    # target_face = dnn_model.forward()
    target_detected = None
    # for i in range(0, target_face.shape[2]):
    #     confidence = target_face[0, 0, i, 2]
    #     if confidence >= 0.8:
    #         box = target_face[0, 0, i, 3:7] * np.array([target_img.shape[0], target_img.shape[1], target_img.shape[0], target_img.shape[1]])
    #         (x, y, w, h) = box.astype(int)
    #         target_captured = square_img(target_img[int(y):int(y + h), int(x):int(x + w)])
    #         target_detected = cv2.resize(target_captured, (160, 160))

    for (x, y, w, h) in target_face:
        target_detected = cv2.resize(target_img[int(y):int(y + h), int(x):int(x + w)], (160, 160))
    target_detected = cv2.cvtColor(target_detected, cv2.COLOR_GRAY2BGR)
    target_restored = np.expand_dims(target_detected, axis=0)
    target_prediction = model.predict(target_restored)

    duration, detection = [], []
    num = 0
    while True:
        success, img = vid.read()
        if not success:
            break
        # elif num >= 300:
            # writer.release()
        #     break
        face = cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.3, 5)
        # blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        # dnn_model.setInput(blob)
        # face = dnn_model.forward()
        # (l, w) = img.shape[:2]
        print('\r' + "Analyzing frame " + str(num) + " of " + str(length) + "   ", end='')
        num += 1

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
                    detected = temp_detected
                    detected_cdn = (x, y, w, h)

        if detected is None or msc > 60:
            detection.append(0)
            duration.append(-0.01)
            text = "Detection: 0"
            cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.imwrite(out_dir + "/" + str(num) + ".jpg", img)
        else:
            detection.append(1)
            shape = predictor(detected, dlib.rectangle(int(0), int(0), int(160), int(160)))

            i = 0
            (x, y, w, h) = detected_cdn
            left, right, top, bottom = 0, 0, 0, 0
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            for pt in shape.parts():
                i += 1
                if i >= 49:
                    cv2.circle(img, (pt.x + x, pt.y + y), 2, (0, 255, 0), 1)
                    if i == 61:
                        left = pt
                    elif i == 65:
                        right = pt
                    elif i == 63:
                        top = pt
                    elif i == 67:
                        bottom = pt

            rate = dist(top, bottom) / dist(left, right)
            duration.append(rate)
            text = "Detection: 1; Distance: " + str(rate)
            if rate >= 0.05:
                cv2.putText(img, text, (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
            else:
                cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.imwrite(out_dir + "/" + str(num) + ".jpg", img)
        # writer.write(img.astype("uint8"))

    # writer.release()
    return duration, detection, length

def calcSpeed(duration, detection):
    acc = []
    for i in range(0, len(duration)):
        front, back = 0, 0
        if i == len(duration)-1:
            front = duration[i] + (duration[i] - duration[i-1])
        else:
            front = duration[i+1]
        if i == 0:
            back = duration[i] - (duration[i] - duration[i-1])
        else:
            back = duration[i-1]
        acc.append(front - back)
    return acc

def main():
    # filepath = "E:\\Participant #1\\videos\\Week 2\\Day 5\\" # "database\\"#
    filepath = "D:\\Participants\\"
    filename = r"P2_Week1_Friday_7-June-2019_02_FACE" # "matt-in-the-lab-test_FACE"
    profile = r"database/P2.jpg"
    filedir = filepath + filename
    length = 0
    '''
    if not os.path.exists(filename + ".txt"):
        duration, detection, length = analyze_video(filedir, profile)
        acc = calcSpeed(duration, detection)
        f = open(filename + ".txt", "w")
        f.write(",".join([str(_) for _ in duration]))
        f.write("\n")
        f.write(",".join([str(_) for _ in detection]))
        f.flush()
        f.close()
    else:
        f = open(filename + ".txt", "r")
        lines = f.readlines()
        duration = [float(_) for _ in lines[0].split(",")]
        detection = [bool(_) for _ in lines[1].split(",")]
        length = len(duration)
        # acc = calcSpeed(duration, detection)
    '''
    duration, detection, length = analyze_video(filedir, profile)
    # acc = calcSpeed(duration, detection)
    b, a = signal.butter(1, 0.1)
    duration_raw = copy.deepcopy(duration)
    duration = signal.filtfilt(b, a, duration)
    acc = calcSpeed(duration, detection)
    # inst = process_alt(duration, acc, bias=0.005, min_speed=0.01, length=length)
    inst = process(duration, acc, min_dist=0.07, min_speed=0.05)
    print('')
    for ii in inst:
        print("From frame " + str(ii[0]) + " to " + str(ii[1]) + ": subject at state " + str(ii[2]) + ";")

    det = []
    for ii in inst:
        for i in range(ii[0], ii[1]):
            det.append(ii[2]/3)

    xl = [float(_/30) for _ in range(0, length)]

    '''
    const = [4, 7, 10, 17, 20, 29, 34, 38, 47, 56, 58, 60, 65, 71, 77, 102, 106, 114]
    con_i = []
    ptr = 0
    for time in xl:
        if float(const[ptr]) == time:
            con_i.append(-0.05)
            ptr += 1
        else:
            con_i.append(-0.1)
    '''

    fig = plt.figure(num=None, figsize=(20, 8), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(xl, duration, "-o", linewidth=0.5, markersize=0.8)
    ax.plot(xl, acc, "-o", linewidth=0.5, markersize=0.8)
    ax.plot(xl, det, linewidth=0.5)
    # ax.plot(xl, con_i, linewidth=0.5)
    plt.show()
    # f.close()


if __name__ == "__main__":
    main()

# 4, 7, 9, 10-11, 17, 20, 29-30, 34-35, 38, 47-49(?), 56-57, 59, 60-61, 65, 71, 77, 102, 106, 114