import cv2 # type: ignore

def list_video_devices():
    print("\n📷 Video Device Scanner")
    print("=======================")
    
    # Check first 10 indexes
    for index in range(10):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Get resolution
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"✅ Index {index}: Found Camera/Capture Card")
                print(f"   Resolution: {int(width)}x{int(height)} @ {fps} FPS")
            else:
                print(f"⚠️ Index {index}: Opened but failed to read frame")
            cap.release()
        else:
            pass # No device at this index

    print("\nNote: 'Cam Link 4K' is likely one of the indexes showing 1920x1080 or 3840x2160.")

if __name__ == "__main__":
    list_video_devices()