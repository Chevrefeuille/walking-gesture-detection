import cv2

def plot_coords(event, x, y, flags, param):
    """
    Mouse callback to print the coords where the mouse was clicked
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        c = '(' + str(2*x) + ', ' + str(2*y) + ')'
        cv2.circle(frame, (x, y), 3, (0,0,255), -1)
        cv2.putText(
            frame, c, (x + 10, y + 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255)
        )

dyad_cap = cv2.VideoCapture('./videos/00000_dyads/sub_1.avi')

# read first frame
ret, frame = dyad_cap.read()

frame = cv2.resize(frame, dsize=(0, 0), fx=1/2, fy=1/2)
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", plot_coords)

while True:
    # display the  frame
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.waitKey(0)
dyad_cap.release()
cv2.destroyAllWindows()
