from PIL import ImageDraw
from PIL import Image

def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]

def _draw_boxes(image, boxes, box_number):
    draw = ImageDraw.Draw(image)
    for i, box in enumerate(boxes):
        if box_number and i == box_number:
            break
        draw.rectangle(unnormalize_box(box, image.width, image.height), outline='red', width=2)
    return image

def draw_boxes(example, box_number=None):
    image = Image.open(example['image_path'])
    _draw_boxes(image, example['bboxes'], box_number).show()
