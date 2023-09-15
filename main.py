import argparse
from viewer.video_viewer import VideoViewer

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id")
    parser.add_argument("-f", "--folder", help="The folder to read videos and labels from.", default="labeled")
    args = parser.parse_args()
    VideoViewer(args.folder, args.video_id).show_video()
