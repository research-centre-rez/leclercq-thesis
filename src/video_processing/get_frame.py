import os
import cv2 as cv
import argparse

argparser = argparse.ArgumentParser()

argparser.add_argument('--vp', type=str, help='Relative video path to the video you want to extract from')
argparser.add_argument('--fn', type=int, help='Frame number you want to extract (int)')
argparser.add_argument('--save', action='store_true')

def get_frame_from_video(vid_path, frame_number, save=False):
    '''
    Returns a specific frame from a video, can also save it if required. The saving name scheme is the following: {name_of_video}_{frame_number}. 
    '''
    base_name  = os.path.splitext(os.path.basename(vid_path))[0]
    
    cap = cv.VideoCapture(vid_path)
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    if frame_number > length:
        print(f"Frame number {frame_number} should not exceed {length}!")
        return 0

    cap.set(cv.CAP_PROP_POS_FRAMES, frame_number - 1)
    ret, frame = cap.read()

    if ret:
        if save:
            out_path = f'{base_name}_{frame_number}.jpg'
            cv.imwrite(out_path, frame)
            print(f'Image saved as {out_path}')
        cap.release()

        return frame

    print('Did not manage to retrieve the frame')
    return 0

if __name__ == "__main__":
    args = argparser.parse_args()
    get_frame_from_video(vid_path=args.vp, frame_number=args.fn, save=args.save)
