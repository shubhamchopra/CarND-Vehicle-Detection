import pickle
import logging
from utils import *
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import glob
import itertools

class WindowHistory:
    def __init__(self, history_size, threshold):
        self.history = []
        self.history_size = history_size
        self.threshold = threshold

    def addBoxes(self, boxList):
        if len(self.history) == self.history_size:
            # drop the oldest box list and append the latest
            self.history = self.history[1:]
            self.history.append(boxList)
        elif len(self.history) < self.history_size:
            self.history.append(boxList)
        else:
            raise AssertionError("History cannot be more than 10")

    def getWindows(self, img):
        if len(self.history) < self.history_size:
            return []
        else:
            boxes = list(itertools.chain.from_iterable(self.history))
            return merge_detected_windows(img, box_list=boxes, threshold=self.threshold)

class CarDetector:
    def __init__(self, modelFile):
        with open(modelFile, "rb") as f:
            data = pickle.load(f)
            self.model = data["estimator"]
            self.color_space = data["color_space"]
            self.spatial_size = data["spatial_size"]
            self.hist_bins = data["hist_bins"]
            self.orient = data["orient"]
            self.pix_per_cell = data["pix_per_cell"]
            self.cell_per_block = data["cell_per_block"]
            self.hog_channel = data["hog_channel"]
        self.ystart = 0.5
        self.ystop = 0.9
        self.threshold = 2
        self.scales = [1.5]
        self.cells_per_step=2
        self.history_size = 10
        self.history = WindowHistory(self.history_size, 0.7*self.threshold*len(self.scales)*self.history_size)
        logging.info("Model loaded")

    def find_car_windows(self, img):
        imgY = img.shape[0]
        windows = []
        for scale in self.scales:
            boxes = find_cars(img, np.int(imgY*self.ystart), np.int(imgY*self.ystop), self.cells_per_step,
                              scale, self.model, self.orient, self.color_space,
                              self.pix_per_cell, self.cell_per_block,
                              self.spatial_size, self.hist_bins)
            windows.extend(boxes)

        # logging.info("Found {} boxes.".format(len(windows)))
        self.history.addBoxes(windows)
        return self.history.getWindows(img)

    def process_image(self, img):
        bgrImg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        windows = self.find_car_windows(bgrImg)
        # logging.info("Num windows detected : {}".format(len(windows)))
        image = np.copy(img)
        for window in windows:
            cv2.rectangle(image, window[0], window[1], (0, 255, 255), 6)
        return image

    def process_video(self, inputVid, outputVid):
        clip1 = VideoFileClip(inputVid)
        laneClip = clip1.fl_image(self.process_image)
        laneClip.write_videofile(outputVid, audio=False)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    mainRunner = CarDetector("SVCModel-YCrCb.p")
    input = "test_video.mp4"
    output = "test_video_output.mp4"
    mainRunner.process_video(input, output)
    # for file in glob.glob("test_images/*.jpg"):
    #     img = mpimg.imread(file)
    #
    #     ret = mainRunner.process_image(img)
    #     outputfile = "output_images/" + file
    #     mpimg.imsave(outputfile, ret)
