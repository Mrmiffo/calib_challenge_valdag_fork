import cv2
import numpy
import math

LINE_COLOR = (0,0,255)
LINE_WIDTH = 3
LINE_LENGTH = 750
PITCH_IND_CENTER = (100, 100)
YAW_IND_CENTER = (300, 100)
IND_COLOR = (255,0,0)

class VideoViewer():

    def __init__(self, folder, name) -> None:
        self.folder = folder
        self.video_name = name
        self.base_path = self.folder+"/"+self.video_name

    def read_labels(self):
        """ Reads the labels from files. Returning a list of tuples, one for each frame, with the pitch and yaw angle in radians."""
        with open(self.base_path+".txt") as f:
            frame_labels = f.read().splitlines()
        return ([(float(f[0]), float(f[1])) for f in [fs.split(" ", 2) for fs in frame_labels]])

    def frame_rad_to_point(self, frame_label):
        """ Converts the pitch and yaw radians to points on a unit circle. Return a tuple with pitch and yaw as ((pitch_x, pitch_y), (yaw_x, yaw_y))"""
        pitch = (math.cos(frame_label[0]), math.sin(frame_label[0]))
        yaw = (math.cos(frame_label[1]), math.sin(frame_label[1]))
        return (pitch, yaw)
    
    def add_indicator(self, frame, center, current_value, name):
        IND_SIZE = 100
        """ Adds a small unit circle graph to the frame indicating the value (pitch, or yaw) provided. """
        frame = cv2.line(frame, center, (center[0]+IND_SIZE, center[1]), color=IND_COLOR, thickness=1)
        frame = cv2.line(frame, center, (center[0]-IND_SIZE, center[1]), color=IND_COLOR, thickness=1)
        frame = cv2.line(frame, center, (center[0], center[1]+IND_SIZE), color=IND_COLOR, thickness=1)
        frame = cv2.line(frame, center, (center[0], center[1]-IND_SIZE), color=IND_COLOR, thickness=1)
        frame = cv2.line(frame, center, (center[0]+int(current_value[0]*IND_SIZE), center[1]-int(current_value[1]*IND_SIZE)), color=LINE_COLOR, thickness=2)
        frame = cv2.putText(frame, name, (center[0]-20, center[1]+115), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=IND_COLOR)
        return frame

    
    def show_video(self, show_label=True):
        # Create a VideoCapture object and read from input file
        cap = cv2.VideoCapture(self.base_path+".hevc")

        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            raise Exception("Error opening video stream or file")
            
        if show_label:
            frame_labels = self.read_labels()

 
        # Read until video is completed
        print("Opening recording in seperate window. Press 'q' to exit, 'p' to pause.")
        frame_id = 0
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
            
                if show_label:
                    # Get the label of the current frame
                    frame_label = frame_labels[frame_id]
                    # Don't try to show a line if there are no labels for the frame.
                    if not math.isnan(frame_label[0]) and not math.isnan(frame_label[1]):
                        center_screen = (int(frame.shape[1]/2), int(frame.shape[0]/2))

                        # Convert the pitch and yaw radians to points on a unit circle
                        pitch_point, yaw_point = self.frame_rad_to_point(frame_label)
                        # Take the y coordinate of pitch to be the X coordinate and yaw y coordinate to be the Y coordinate
                        # Increase the length of the line arbitrarily
                        target_point = (int(yaw_point[1]*LINE_LENGTH), int(pitch_point[1]*LINE_LENGTH))
                        final_target = (center_screen[0]+target_point[0], center_screen[1]-target_point[1])

                        # Add the line to the frame indicating the pitch and yaw from the center of the screen
                        frame = cv2.line(frame, center_screen, final_target, LINE_COLOR, LINE_WIDTH)
                        frame = cv2.circle(frame, center_screen, radius=5, color=LINE_COLOR, thickness=-1)
                        frame = self.add_indicator(frame, PITCH_IND_CENTER, pitch_point, "Pitch")
                        frame = self.add_indicator(frame, YAW_IND_CENTER, yaw_point, "Yaw")
                # Display the resulting frame
                cv2.imshow(self.video_name,frame)
            
                key = cv2.waitKey(20)
                if key == ord('q'):
                    # "Press 'q' to quit
                    break
                if key == ord('p'):
                    # Press 'p' to Pause
                    cv2.waitKey(-1) #wait until any key is pressed
            
            # Break the loop
            else: 
                break
            frame_id += 1
 
        # When everything done, release the video capture object
        cap.release()
 
        # Closes all the frames
        cv2.destroyAllWindows()