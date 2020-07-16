import numpy as np
import argparse, time
import cv2
from imutils.video import VideoStream
import imutils

model = 'res10_300x300_ssd_iter_140000.caffemodel'
prototxt = 'deploy.prototxt.txt'
confidence_in = 0.8  # default

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inputtype", required=True, help="video/image")
ap.add_argument("-im", "--image", required=False, default=None, help="path to input image")
args = vars(ap.parse_args())

# load our serialized model from disk
net = cv2.dnn.readNetFromCaffe(prototxt, model)

input_type = args["inputtype"]

if input_type == "image":
    # load the input image and construct an input blob for the image by resizing to a fixed 300x300 pixels and then normalizing it
    image = cv2.imread(args["image"])
    # if not image:
    #     print("No image path supplied!")
    #     exit()
 
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_in:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the bounding box of the face along with the associated probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    
elif input_type == "video":
    # initialize the video stream and allow the camera sensor to warm up
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame and resize it to a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]
            if confidence < confidence_in:
                continue

            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the bounding box of the face along with the associated probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
