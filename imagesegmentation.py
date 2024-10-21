from __future__ import division
import cv2
import numpy as np
import os
import sys
import argparse
from time import time
from math import exp, pow
from augmentingPath import augmentingPath
from pushRelabel import pushRelabel

# np.set_printoptions(threshold=np.inf)
graphCutAlgo = {"ap": augmentingPath,
                "pr": pushRelabel
                }
SIGMA = 30
# LAMBDA = 1
OBJCOLOR, BKGCOLOR = (0, 0, 255), (0, 255, 0)
OBJCODE, BKGCODE = 1, 2
OBJ, BKG = "OBJ", "BKG"

CUTCOLOR = (255, 0, 0)

SOURCE, SINK = -2, -1
SF = 10 # Scale Factor
LOADSEEDS = False
# drawing = False

def show_image(image):
    window_name = "Segmentation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plantSeed(image):

    def drawLines(x, y, pixelType):
        if pixelType == OBJ:
            color, code = OBJCOLOR, OBJCODE
        else:
            color, code = BKGCOLOR, BKGCODE
        cv2.circle(image, (x, y), radius, color, thickness)
        cv2.circle(seeds, (x // SF, y // SF), radius // SF, code, thickness)

    def onMouse(event, x, y, flags, pixelType):
        global drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            drawLines(x, y, pixelType)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            drawLines(x, y, pixelType)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    def paintSeeds(pixelType):
        print("Planting", pixelType, "seeds")
        global drawing
        drawing = False
        window_name = "Plant " + pixelType + " seeds"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, onMouse, pixelType)
        while (1):
            cv2.imshow(window_name, image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()


    seeds = np.zeros(image.shape, dtype="uint8")
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (0, 0), fx=SF, fy=SF)

    radius = 5
    thickness = -1 # fill the whole circle
    global drawing
    drawing = False

    paintSeeds(OBJ)
    paintSeeds(BKG)
    return seeds, image


# Large when ip - iq < sigma, and small otherwise
def boundaryPenalty(ip, iq):
    bp = 100 * exp(- pow(int(ip) - int(iq), 2) / (2 * pow(SIGMA, 2)))
    return bp

def buildGraph(image):
    V = image.size + 2
    graph = np.zeros((V, V), dtype='int32')
    K = makeNLinks(graph, image)
    seeds, seededImage = plantSeed(image)
    makeTLinks(graph, seeds, K)
    return graph, seededImage

def makeNLinks(graph, image):
    K = -float("inf")
    r, c = image.shape
    for i in range(r):
        for j in range(c):
            x = i * c + j
            if i + 1 < r: # pixel below
                y = (i + 1) * c + j
                bp = boundaryPenalty(image[i][j], image[i + 1][j])
                graph[x][y] = graph[y][x] = bp
                K = max(K, bp)
            if j + 1 < c: # pixel to the right
                y = i * c + j + 1
                bp = boundaryPenalty(image[i][j], image[i][j + 1])
                graph[x][y] = graph[y][x] = bp
                K = max(K, bp)
    return K



def makeTLinks(graph, seeds, K):
    r, c = seeds.shape

    for i in range(r):
        for j in range(c):
            x = i * c + j
            if seeds[i][j] == OBJCODE:
                # graph[x][source] = K
                graph[SOURCE][x] = K
            elif seeds[i][j] == BKGCODE:
                graph[x][SINK] = K
                # graph[sink][x] = K
            # else:
            #     graph[x][source] = LAMBDA * regionalPenalty(image[i][j], BKG)
            #     graph[x][sink]   = LAMBDA * regionalPenalty(image[i][j], OBJ)



def displayCut(image, cuts):
    def colorPixel(i, j):
        image[i][j] = CUTCOLOR

    r, c = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # return image
    # Define blue color with RGB (B, G, R) and opacity 60%
    BLUE_COLOR = (255, 0, 0)  # OpenCV uses BGR format
    OPACITY = 0.6

    # Convert image to RGB if it's grayscale
    if len(image.shape) == 2:  # Check if the image is grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Create a copy of the original image to draw the polygon with transparency
    overlay = image.copy()

    # Convert cuts list into points suitable for drawing polygon
    polygon_points = []
    r, c = image.shape[:2]  # Get the dimensions of the image
    for cut in cuts:
        if cut[0] != SOURCE and cut[0] != SINK and cut[1] != SOURCE and cut[1] != SINK:
            point1 = (cut[0] % r, cut[0] // r)
            point2 = (cut[1] % r, cut[1] // r)
            colorPixel(cut[0] // r, cut[0] % r)
            colorPixel(cut[1] // r, cut[1] % r)
            polygon_points.extend([point1, point2])
    print(polygon_points)
    if polygon_points:
        # Convert list of points into a format suitable for fillPoly
        polygon_points = np.array([polygon_points], dtype=np.int32)

        # Draw filled polygon on the overlay with the desired color
        cv2.fillPoly(overlay, polygon_points, BLUE_COLOR)

        # Blend the overlay with the original image using the specified opacity
        image = cv2.addWeighted(overlay, OPACITY, image, 1 - OPACITY, 0)

    return image



def imageSegmentation(imagefile, size=(30, 30), algo="ap"):
    pathname = os.path.splitext(imagefile)[0]
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, size)
    graph, seededImage = buildGraph(image)
    cv2.imwrite(pathname + "seeded.jpg", seededImage)

    global SOURCE, SINK
    SOURCE += len(graph)
    SINK   += len(graph)

    cuts = graphCutAlgo[algo](graph, SOURCE, SINK)
    print("cuts:")
    print(cuts)
    image = displayCut(image, cuts)
    image = cv2.resize(image, (0, 0), fx=SF, fy=SF)
    show_image(image)
    save_name = pathname + "_cut.jpg"
    cv2.imwrite(save_name, image)
    print("Saved image as", save_name)


def parseArgs():
    def algorithm(string):
        if string in graphCutAlgo:
            return string
        raise argparse.ArgumentTypeError(
            "Algorithm should be one of the following:", graphCutAlgo.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument("imagefile")
    parser.add_argument("--size", "-s",
                        default=50, type=int,
                        help="Defaults to 30x30")
    parser.add_argument("--algo", "-a", default="ap", type=algorithm)
    return parser.parse_args()

if __name__ == "__main__":

    args = parseArgs()
    start = time()
    imageSegmentation(args.imagefile, (args.size, args.size), args.algo)
    elapsed_time = time() - start
    print(f"Time taken: {elapsed_time/60:.2f} minutes")
