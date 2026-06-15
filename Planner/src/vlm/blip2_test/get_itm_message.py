import sys
import os
import time
# Add project root to sys.path so absolute imports work when running this script directly
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np
from vlm.blip2_test.test_blip2 import BLIP2ITMClient

itmclient = BLIP2ITMClient(port=12182)

def get_itm_message(rgb_image, label):
    txt = f"Is there a {label} in the image?"
    cosine = itmclient.cosine(rgb_image, txt)
    itm_score = itmclient.itm_score(rgb_image, txt)
    return cosine, itm_score

def get_itm_message_cosine(rgb_image, label, room):
    if room != "everywhere":
        txt = f"Seems like there is a {room} or a {label} ahead?"
    else:
        txt = f"Seems like there is a {label} ahead?"
    start_time = time.time()
    cosine = itmclient.cosine(rgb_image, txt)
    end_time = time.time()
    print(f"Time taken for cosine similarity calculation: {end_time - start_time:.4f} seconds")
    return cosine

if __name__ == "__main__":
    rgb_image = cv2.imread(os.path.join(os.path.dirname(__file__), "img2.png"))
    # cv2.imshow("Test Image", rgb_image)
    # cv2.waitKey(0)
    label = "bed"
    room = "livingroom"
    # cosine, itm_score = get_itm_message(rgb_image, label)
    # print(f"Cosine similarity: {cosine}, ITM score: {itm_score}")
    cosine_only = get_itm_message_cosine(rgb_image, label, room)
    print(f"Cosine similarity only: {cosine_only}")
    # cv2.destroyAllWindows()