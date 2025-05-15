import cv2
import numpy as np
import time

# Constant for the marge of obstacles
SIZE_ROBOT = 65

def is_red(color):
    b, g, r = color
    return (float(r) - float(b)) > 50 and (float(r) - float(g)) > 50

def is_green(color):
    b, g, r = color
    return (float(g) - float(b)) > 20 and (float(g) - float(r)) > 20

def is_black(color):
    b, g, r = color
    return np.abs(float(b) - float(r)) < 20 and np.abs(float(g) - float(r)) < 20 and np.abs(float(g) - float(b)) < 20 and r<70 and b<70 and g<70


def detect_obstacles(frame):
    colors = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3 , 3), 0)
    edges = cv2.Canny(blurred, 100, 100)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    black_rectangles = []
    green_rectangles = []
    red_rectangles = []
    
    for i, contour in enumerate(contours):
        # approximation of the shape of contour as a polygone
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        # keep only the pixels inside the polygone
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        inside_contour = cv2.bitwise_and(colors, colors, mask=mask)
        pixels = inside_contour[mask == 255]
        

        mean_color = np.mean(pixels, axis=0)

        if len(approx) == 4:  # rectangles has 4 sides
            x, y, w, h = cv2.boundingRect(contour)
            if is_black(mean_color):
                margin_rect = (x - SIZE_ROBOT, y - SIZE_ROBOT, x + w + SIZE_ROBOT, y + h + SIZE_ROBOT)
                black_rectangles.append(margin_rect)
                cv2.rectangle(frame, (margin_rect[0], margin_rect[1]), (margin_rect[2], margin_rect[3]), (255, 255, 255), 2)
            # w>20 is used to not take in consideration the leds of the thymio
            elif is_red(mean_color) and w>20 :
                margin_rect = (x - SIZE_ROBOT, y - SIZE_ROBOT, x + w + SIZE_ROBOT, y + h + SIZE_ROBOT)
                red_rectangles.append(margin_rect)
                cv2.rectangle(frame, (margin_rect[0], margin_rect[1]), (margin_rect[2], margin_rect[3]), (0, 0, 255), 2)
            elif is_green(mean_color):
                green_rectangles.append((x, y, x + w, y + h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame,  black_rectangles, green_rectangles, red_rectangles


def detect_thymio(frame):
    colors = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thres = cv2.threshold(gray,90,255,cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(thres, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 50)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blue_center = None
    blue_orientation_angle = None
    
    for i, contour in enumerate(contours):
        # approximation of the shape of contour as a polygone
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        # keep only the pixels inside the polygone
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        inside_contour = cv2.bitwise_and(colors, colors, mask=mask)
        pixels = inside_contour[mask == 255]

        mean_color = np.mean(pixels, axis=0)
        
        if len(approx) == 3 and is_black(mean_color):  # triangles have 3 sides
            
            # compute the moments to define the center
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                blue_center = (cx, cy)
                # cv2.circle(frame, blue_center, 5, (255, 0, 0), -1)

            # approximation of the vertices of the triangle
            vertices = approx.reshape((3, 2))
            
            # distances between each vertice and the center
            distances = [np.linalg.norm(np.array(blue_center) - vertex) for vertex in vertices]
            
            # take the furthest (isoscele triangle) 
            max_index = np.argmax(distances)
            max_vertice = vertices[max_index]

            # compute angle with horizontal
            dx = max_vertice[0] - blue_center[0]
            dy = blue_center[1] - max_vertice[1]  # invert because origin in top left
            blue_orientation_angle = np.arctan2(dy, dx) 

            # cv2.line(frame, blue_center, tuple(max_vertice), (255, 0, 0), 2)

    return frame, blue_center, blue_orientation_angle


def sheet(frame, cap):
    while True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thres = cv2.threshold(blurred,100,255,cv2.THRESH_BINARY)
        edges = cv2.Canny(thres, 50, 50)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            if w > 200 and h > 200:
                return x,y,w,h
            else :
                _, frame = cap.read()