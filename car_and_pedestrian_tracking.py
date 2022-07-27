import cv2

# video download
video = cv2.VideoCapture('videoplayback.mp4')

# pretrained car and pedestrian classifiers
car_tracker_file = 'car_detector.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'


# create car classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

# Run till driving stops
while True:
    # Read the current frame
    (read_successful, frame) = video.read()

    if read_successful:
        # convert to greyscale to be able to use haar cascade
        greyscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    # detect cars and pedestrians
    cars = car_tracker.detectMultiScale(greyscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(greyscaled_frame)

    # Draw rectangles around the cars and pedestrian
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 4)

    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 4)

    # Display image with car spotted
    cv2.imshow('Hemie\'s Self Driving Car', frame)

    # Wait key: Listen for a key press, do not autoclose
    key = cv2.waitKey()

    # Stop if Q key is pressed (check ASCII code for q and Q) to get out of the loop
    if key == 81 or key == 113:
        break


# to release VideoCapture Object
video.release()


print('Code Complete')
