from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
import cv2
import numpy as np
import imutils

def process_thresholded_image():
    global model, processor
    thresholded_img = cv2.imread('captured_image.jpg', cv2.IMREAD_GRAYSCALE)
    thresholded_img_rgb = cv2.cvtColor(thresholded_img, cv2.COLOR_GRAY2RGB)
    thresholded_img_pil = Image.fromarray(thresholded_img_rgb)
    
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
    pixel_values = processor(images=thresholded_img_pil, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("Generated Text:", generated_text)
def capture_image():
    global img
    cv2.imwrite('captured_image.jpg', img)
    print("Image captured and saved as captured_image.jpg")
    process_thresholded_image()

url = "http://100.99.247.151:8080/shot.jpg"
while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = imutils.resize(img, width=1000, height=1800)
    cv2.imshow("Android_cam", img)
    key = cv2.waitKey(1)
    if key == 27: 
        break
    elif key == ord('c'):
        capture_image()

cv2.destroyAllWindows()
