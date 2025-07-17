import cv2
import pytesseract
import re

def process_receipt(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    text = pytesseract.image_to_string(thresh)

    lines = text.split('\n')
    items = []
    for line in lines:
        match = re.match(r'(.*?)(\d+\.\d{2})$', line.strip())
        if match:
            name = match.group(1).strip()
            price = match.group(2).strip()
            items.append((name, price))

    return items