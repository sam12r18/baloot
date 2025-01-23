import cv2
from PIL import Image
from transparent_background import Remover
import torch
# from RealESRGAN import RealESRGAN
from PIL import ImageDraw, ImageOps
import dlib
import numpy as np
from background_remove.faceorienter.faceorienter import FaceOrienter
#import face_recognition
import inspect
from retinaface import RetinaFace
import uuid
import time
import multiprocessing

def metric_to_pixel(value, metric = 'px', dpi=300):
        if(metric == 'mm'):
            return round((value * dpi) / 25.4)
        if(metric == 'cm'):
            return round((value * dpi) / 2.54)
        if metric == 'in':
            return round(value * dpi)
        return value


class ImageProcessor:
    def __init__(self):
        self.remover = Remover(ckpt='latest.pth', device="cpu")
        self.device = torch.device('cpu')
        self.model = RealESRGAN(self.device, scale=4)
        self.model.load_weights('background_remove/transparent-background/weights/RealESRGAN_x4.pth', download=True)

    def remove_background(self, image_path):
        img = Image.open(image_path).convert('RGB')
        out = self.remover.process(img, type='white')
        output_path = 'bg_remove.jpg'
        Image.fromarray(out).save(output_path)
        return output_path

    def upscale_image(self, image_path):
        path_to_image = image_path
        image = Image.open(path_to_image)
        sr_image = self.model.predict(image)
        output_path = 'upscaled_sample.jpg'
        sr_image.save(output_path)
        return output_path

    # def fix_orientation(self, image_path):
    #     fo = FaceOrienter(image_path)
    #     orientation = fo.predict_orientation()
    #     fixed_image_path = 'fixed_orientation_sample.jpg'
    #     fo.fix_orientation(fixed_image_path)
    #     return fixed_image_path


    def find_chin_height(self, img):

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # Load the image
        #img = cv2.imread("sam05.jpg")

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = detector(gray)


        max_y = -float('inf')
        max_y_index = None

        # Loop through each face found
        for face in faces:
            # Detect facial landmarks
            landmarks = predictor(gray, face)

            # Loop through each landmark point
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y

                # Update maximum y-coordinate and its corresponding landmark index if a higher y-coordinate is found
                if y > max_y:
                    max_y = y
                    max_y_index = n

        return max_y


    def find_top_head(self, img):
        # Convert the image to grayscale
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 20, 50)

        # Find contours in the edge image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Merge all contour arrays into a single array
        merged_contours = np.concatenate(contours)

        return merged_contours[-1][0][1]


    def fix_face_angle(self, img):
        faces = RetinaFace.detect_faces(np.array(img))
        face_landmarks = faces['face_1']['landmarks']

        if len(face_landmarks) == 0:
            print("No face detected in the image.")
            return None

        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']
        dx = left_eye[0] - right_eye[0]
        dy = left_eye[1] - right_eye[1]

        angle = np.degrees(np.arctan2(dy, dx))
        return img.rotate(angle, expand=True, fillcolor='white')

    def find_eye_to_eye(self, img):
        faces = RetinaFace.detect_faces(img)
        face_landmarks = faces['face_1']['landmarks']

        if len(face_landmarks) == 0:
            print("No face detected in the image.")
            return None

        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']
        dx = left_eye[0] - right_eye[0]
        dy = left_eye[1] - right_eye[1]

        return dx


    def find_eye_height(self, img):
        faces = RetinaFace.detect_faces(img)
        face_landmarks = faces['face_1']['landmarks']

        if len(face_landmarks) == 0:
            print("No face detected in the image.")
            return None

        left_eye = face_landmarks['left_eye']

        return left_eye[1]

    def rotate_and_resize_face(self, image_path, face_height, distance_top, eye_to_eye, eye_to_bottom, eye_to_chin, metric='px'):

        face_height = metric_to_pixel(face_height, metric)
        distance_top = metric_to_pixel(distance_top, metric)
        eye_to_bottom = metric_to_pixel(eye_to_bottom, metric)
        eye_to_eye = metric_to_pixel(eye_to_eye, metric)
        eye_to_chin = metric_to_pixel(eye_to_chin, metric)

        image = Image.open(image_path).convert('RGB')

        rotated_image = self.fix_face_angle(image)

        top_head = self.find_top_head(np.array(rotated_image))

        ratio_to_resize_head = 0
        if(eye_to_eye):
            current_eye_to_eye = self.find_eye_to_eye(np.array(rotated_image))
            if current_eye_to_eye == 0:
                print("Invalid face landmarks. Could not calculate the current distance.")
                return None
            ratio_to_resize_head = (eye_to_eye) / current_eye_to_eye
        elif(eye_to_chin):
          chin_height = self.find_chin_height(np.array(rotated_image))
          eye_height = self.find_eye_height(np.array(rotated_image))
          current_chin_to_eye = chin_height - eye_height
          if current_chin_to_eye == 0:
              print("Invalid face landmarks. Could not calculate the current distance.")
              return None
          ratio_to_resize_head = (eye_to_chin) / current_chin_to_eye
        else:
            chin_height = self.find_chin_height(np.array(rotated_image))
            current_face_height = chin_height - top_head
            if current_face_height == 0:
                print("Invalid face landmarks. Could not calculate the current distance.")
                return None
            ratio_to_resize_head = (face_height) / current_face_height



        # Resize the rotated image based on the ratio
        new_width = int(rotated_image.width * ratio_to_resize_head)
        new_height = int(rotated_image.height * ratio_to_resize_head)

        resized_image = rotated_image.resize((new_width, new_height))


       # Convert the image to grayscale
        gray = cv2.cvtColor(np.array(resized_image), cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 20, 50)

        # Find contours in the edge image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Merge all contour arrays into a single array
        merged_contours = np.concatenate(contours)

        top_head = merged_contours[-1][0][1]

        faces2 = RetinaFace.detect_faces(np.array(resized_image))

        landmarks2 = faces2['face_1']['landmarks']


        nose_tip_x = landmarks2['nose'][0]
        nose_tip_y = landmarks2['nose'][1]
        image_width, image_height = resized_image.size
        target_x = image_width // 2

        if (eye_to_bottom):
          current_right_eye_height = landmarks2['left_eye'][1]
          translate_y = ((image_height - int(current_right_eye_height)) - (eye_to_bottom))
        else:
          translate_y = (distance_top - top_head)


        translate_x = target_x - int(nose_tip_x)


        # Apply translation to the rotated image
        translated_image = Image.new('RGB', (image_width, image_height), 'white')
        translated_image.paste(resized_image, (int(translate_x), int(translate_y)))



        rotate_and_resize_face_path = 'rotate_and_resize_face.jpg'
        translated_image.save(rotate_and_resize_face_path, format='JPEG', quality=100)
        return rotate_and_resize_face_path


    def resize_and_shift_image(self, image_path, target_width, target_height, top_cut, shift_x_pixels=0, shift_y_pixels=0, metric="px"):


        target_width = metric_to_pixel(target_width, metric)
        target_height = metric_to_pixel(target_height, metric)

        # Open the image
        image = Image.open(image_path)

        # Calculate the current width and height
        current_width, current_height = image.size

        # Calculate the difference between target and current width and height
        width_diff = current_width - target_width
        height_diff = current_height - target_height

        if width_diff > 0:
            # Cut the extra width equally from both sides
            left_cut = width_diff // 2
            right_cut = width_diff - left_cut
            image = image.crop((left_cut + shift_x_pixels, 0, current_width - (right_cut - shift_x_pixels), current_height))
        elif width_diff < 0:
            # Add extra width equally on both sides as white pixels
            left_pad = abs(width_diff) // 2
            right_pad = abs(width_diff) - left_pad
            image = ImageOps.expand(image, border=(left_pad, 0, right_pad, 0), fill='white')

        if height_diff > 0:
            # Cut the extra height equally from both top and bottom
            if top_cut:
              image = image.crop((0, height_diff, image.width, current_height))
            else:
              image = image.crop((0, 0, image.width, current_height - (height_diff)))

        elif height_diff < 0:
            # Add extra height equally on both top and bottom as white pixels
            pad = abs(height_diff)
            if top_cut:
              image = ImageOps.expand(image, border=(0, pad, 0, 0), fill='white')
            else:
              image = ImageOps.expand(image, border=(0, 0, 0, pad), fill='white')


        # Resize the image to the target dimensions
        resized_image = image.resize((target_width, target_height))


        resize_and_shift_image_path = 'resize_and_shift_image_path.jpg'
        resized_image.save(resize_and_shift_image_path, format='JPEG', quality=100)
        return resize_and_shift_image_path

    def bg_color(self, image_path, color=0):

        output_path = 'result.jpg'
        img = Image.open(image_path).convert('RGB')

        if(color):
          out = self.remover.process(img, type=color)
        else:
          output_path = 'result.png'
          out = self.remover.process(img)


        Image.fromarray(out).save(output_path)
        return output_path

    # def change_brightness(self, image_path, alpha = 1.0, beta = 0):
    #   image = cv2.imread(image_path)

    #   manual_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    #   cv2.imwrite('result.jpg', manual_result)