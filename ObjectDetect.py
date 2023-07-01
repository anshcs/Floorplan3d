import torch 
import cv2
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt
import time 

color_mapping = {
            'room': (255, 0, 0),       # Blue
            'window': (0, 255, 0),     # Green
            'wc': (255, 165, 0),       # Orange
            'toilet': (0, 0, 255),   # red
            'sofa': (0, 255, 255),     # Yellow
            'kitchen': (255, 255, 0),  # Cyan
            'bed': (180, 105, 255),    # Pink
            'storage': (42, 42, 165),  # Brown
            'tv': (128, 128, 128),     # Gray
            'dining_table': (255, 0, 255) , # Magenta
            'Walls': (128, 128, 128)
        }



class ObjectDetection:
    
    def __init__(self,model,img_path) -> None:
        self.model = model
        self.img_path = img_path
    
    def Predtict(self):
        self.myModel = torch.load(f'C:/capstone/3dFloorplan - Copy/Models/{self.model}.pt')
        self.predictions = self.myModel.predict(self.img_path, confidence=40, overlap=30).json()
        return self.predictions
    
    def ViewPredictions(self):
            self.Predtict()
        # Load the image using OpenCV
            image = cv2.imread(self.img_path)
            # image = image.resize((640, 640))
            # Define the color mappings for each class
            
            mask = np.zeros((640, 640), dtype=np.uint8)
            # Iterate through the predictions
            for bounding_box in self.predictions['predictions']:
                x = int(bounding_box['x'])
                y = int(bounding_box['y'])
                width = int(bounding_box['width'])
                height = int(bounding_box['height'])
                confidence = bounding_box['confidence']
                class_name = bounding_box['class']
                x0 = bounding_box['x'] - bounding_box['width'] / 2
                x1 = bounding_box['x'] + bounding_box['width'] / 2
                y0 = bounding_box['y'] - bounding_box['height'] / 2
                y1 = bounding_box['y'] + bounding_box['height'] / 2
                start_point = (int(x0), int(y0))
                end_point = (int(x1), int(y1))
                # Get the color for the current class
                color = color_mapping.get(class_name, (128, 128, 128))  # Green color as default if the class is not in the mapping

                # Draw the bounding box on the image using the corresponding color
                cv2.rectangle(image,  start_point, end_point, color=color, thickness=2)
                cv2.putText(image, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
            # Convert the OpenCV image to PIL format
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Create a BytesIO object to store the image data
            image_stream = io.BytesIO()

            # Save the image with bounding boxes to the BytesIO stream
            image_pil.save(image_stream, format='PNG')

            # Rewind the stream to the beginning
            image_stream.seek(0)

            # Display the image in the notebook
            return Image.open(image_stream).show()

# if __name__ == '__main__':
#     img_path = r'C:\capstone\3dFloorplan - Copy\Capstone-4\valid\images\16_jpg.rf.21cbc6ecfeec59d54a1af55ead68c60c.jpg'
#     wallDetect = ObjectDetection('walldetector',img_path=img_path)
#     roomDetect = ObjectDetection('RoomDetector',img_path)
#     windowDetect = ObjectDetection('windowDetetor',img_path)
#     doorDetect = ObjectDetection('DoorDetector',img_path=img_path)
#     print(roomDetect.Predtict())
#     roomDetect.ViewPredictions()
    
    