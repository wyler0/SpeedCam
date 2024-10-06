import cv2 as cv

def show_cursor_position(event, x, y, flags, param):
    if event == cv.EVENT_MOUSEMOVE:
        print(f"Cursor Position: x={x}, y={y}")

# Load the image
# Start of Selection
# cap = cv.VideoCapture('/Users/wylerzahm/Desktop/output.mp4')
# image = None

# for i in range(40):
#     cap.read()
    
# while image is None:
#     ret, image = cap.read()
# cap.release()

image = cv.imread('/Users/wylerzahm/Desktop/Personal/Projects/speed-cam/utils/experimenting/Screenshot.png')

# Create a window and set the mouse callback
cv.namedWindow('Image')
cv.setMouseCallback('Image', show_cursor_position)

while True:
    cv.imshow('Image', image)
    
    # Break the loop on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
