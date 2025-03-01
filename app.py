from flask import Flask, render_template, request, url_for, send_file
import keras
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

# Add these near the top of your file, with other imports and configurations
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def crop_and_colorize(img_path, model_path1, model_path2):
    """
    Crops and colorizes human figures in an image

    Args:
        img_path: Path to input image
        model_path1: Path to first colorization model
        model_path2: Path to second colorization model
    """
    # Load YOLO pose detection model
    model = YOLO("models/yolo11n-pose.pt")

    # Resize input image to 512x512
    resized_img = cv2.resize(cv2.imread(img_path), (512, 512))
    cv2.imwrite('static/temp/resized.jpg', resized_img)

    # Detect poses in the image
    results = model("static/temp/resized.jpg")
    l = []  # List to store processed image data

    print(len(results[0].keypoints.xy))
    # Process each detected person
    for i, keypoint in enumerate(results[0].keypoints.xy):
        # Convert keypoints to integer coordinates
        xy = np.array(keypoint).astype(int)

        # Extract key body points
        nose_x, nose_y = xy[0]
        left_sh_x, left_sh_y = xy[6]
        right_sh_x, right_sh_y = xy[5]

        # Calculate padding based on shoulder width
        shoulder_width = right_sh_x - left_sh_x
        padding_x = int(shoulder_width * 0.2)
        padding_y = int(shoulder_width * 0.4)

        # Read and process image
        img = cv2.imread('static/temp/resized.jpg')
        numpy_image = np.array(img)

        # Calculate crop boundaries
        img_h, img_w, _ = img.shape
        x_min = max(0, min(left_sh_x, nose_x))
        x_max = min(img_w, right_sh_x)
        y_min = max(0, nose_y-padding_y)
        y_max = min(img_h, max(left_sh_y, right_sh_y))

        # Crop and save individual person
        cropped_image = img[y_min:y_max, x_min:x_max]
        if not os.path.exists(f'static/temp/not_colored'):
            os.makedirs(f'static/temp/not_colored')

        # Save cropped image and colorize it
        cv2.imwrite(f'static/temp/not_colored/{i}.jpg', cropped_image)
        colored_image = colorize_image(
            f'static/temp/not_colored/{i}.jpg', model_path1, model_path2)
        colored_image = cv2.resize(
            colored_image, (x_max-x_min, y_max-y_min), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(
            f'static/temp/not_colored/{i}_colorized.jpg', colored_image)

        # Store bounding box and image path
        bbox = (x_min, y_min, x_max, y_max)
        l.append([f'static/temp/not_colored/{i}_colorized.jpg', bbox])

    return l


def mask(img_path):
    """
    Creates a segmentation mask for the image

    Args:
        img_path: Path to input image
    """
    # Define colors for mask
    BG_COLOR = (0, 0, 0)  # Background color (black)
    MASK_COLOR = (255, 255, 255)  # Mask color (white)

    # Configure image segmentation options
    base_options = python.BaseOptions(
        model_asset_path='models/selfie_multiclass_256x256.tflite')
    options = vision.ImageSegmenterOptions(base_options=base_options,
                                           output_category_mask=True)

    # Create and use image segmenter
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        # Load and segment image
        image = mp.Image.create_from_file(img_path)
        segmentation_result = segmenter.segment(image)
        category_mask = segmentation_result.category_mask

        # Generate mask image
        image_data = image.numpy_view()
        fg_image = np.zeros(image_data.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(image_data.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR

        # Apply mask
        condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
        output_image = np.where(condition, fg_image, bg_image)

        return output_image


def colorize(img_path, model_path):
    """
    Colorizes a single image using specified model

    Args:
        img_path: Path to input image
        model_path: Path to colorization model
    """
    # Load model and prepare image
    model = keras.models.load_model(model_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    test = np.expand_dims(img, axis=0)

    # Generate prediction
    predict = model.predict(test)
    predict = np.rint(predict*255.0).astype(np.uint8)
    predict = np.squeeze(predict, axis=0)

    # Convert to LAB color space
    img = img[:, :, 0]
    img = np.expand_dims(img, axis=-1)
    lab_img = np.concatenate((img, predict), axis=-1)
    rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

    return rgb_img


def blend_images(img1, img2, alpha=0.5):
    """
    Blends two images using LAB color space

    Args:
        img1, img2: Input images to blend
        alpha: Blending factor (0-1)
    """
    # Resize images to match
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Convert to LAB color space
    lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

    # Split channels
    L1, A1, B1 = cv2.split(lab1)
    L2, A2, B2 = cv2.split(lab2)

    # Blend color channels
    A_blended = cv2.addWeighted(A1, alpha, A2, 1 - alpha, 0)
    B_blended = cv2.addWeighted(B1, alpha, B2, 1 - alpha, 0)

    # Merge channels and convert back to BGR
    lab_blended = cv2.merge([L1, A_blended, B_blended])
    blended_img = cv2.cvtColor(lab_blended, cv2.COLOR_LAB2BGR)

    return blended_img


def increase_vibrance(image, strength=1.5):
    """
    Increases image vibrance

    Args:
        image: Input image
        strength: Vibrance increase factor
    """
    # Convert to HSV for color manipulation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = cv2.split(hsv)

    # Boost saturation more in less saturated areas
    saturation_boost = (1 - (S / 255)) * strength
    S = S + (S * saturation_boost)

    # Clip values and convert back to BGR
    S = np.clip(S, 0, 255).astype(np.uint8)
    H = H.astype(np.uint8)
    V = V.astype(np.uint8)
    hsv_vibrant = cv2.merge([H, S, V])
    vibrant_img = cv2.cvtColor(hsv_vibrant, cv2.COLOR_HSV2BGR)

    return vibrant_img


def increase_saturation(image, scale=1.2):
    """
    Increases image saturation uniformly

    Args:
        image: Input image
        scale: Saturation multiplier
    """
    # Convert to HSV and increase saturation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    S = np.clip(S * scale, 0, 255).astype(np.uint8)

    # Convert back to BGR
    hsv_saturated = cv2.merge([H, S, V])
    saturated_img = cv2.cvtColor(hsv_saturated, cv2.COLOR_HSV2BGR)

    return saturated_img


def colorize_image(image_path, model_path1, model_path2):
    """
    Main colorization function that combines multiple techniques

    Args:
        image_path: Path to input image
        model_path1, model_path2: Paths to colorization models
    """
    print(f"model_path1: {model_path1}")
    print(f"model_path2: {model_path2}")

    # Detect objects in image
    model = YOLO('models/yolo11n.pt')
    results = model.predict(image_path)

    # Colorize using both models and blend results
    img1 = colorize(image_path, model_path1)
    img2 = colorize(image_path, model_path2)
    img3 = blend_images(img1, img2)
    img4 = increase_saturation(img3, scale=1.1)
    img5 = increase_vibrance(img4, strength=0.6)

    # Special handling for portrait images
    if model_path1 == 'models/finetuned_unsplash_1500_epoch8.keras':
        return img5

    # Additional processing for non-portrait images
    img6 = cv2.imread(image_path)
    img7 = cv2.resize(
        img5, (img6.shape[1], img6.shape[0]), interpolation=cv2.INTER_CUBIC)
    img8 = blend_images(img6, img7)
    img9 = increase_saturation(img8, scale=1.1)
    img10 = increase_vibrance(img9, strength=0.9)
    cv2.imwrite('static/output.jpg', img10)
    return img10


def desaturate(image):
    """
    Removes color saturation from an image

    Args:
        image: Input image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 0  # Set saturation to 0
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def cleanup_temp_files():
    """
    Removes temporary files created during processing
    """
    TEMP_FOLDER = 'static/temp'
    for filename in os.listdir(TEMP_FOLDER):
        file_path = os.path.join(TEMP_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Error deleting {file_path}: {e}')


# create a simple flask application
app = Flask(__name__)

# define a route


@app.route('/', methods=["GET", "POST"])
def colorize_route():
    """
    Main route handler for the web application
    Handles both initial page load and image processing requests
    """
    # Check if file was uploaded
    if 'file' not in request.files:
        return render_template('index.html')

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Process valid image file
    if file and allowed_file(file.filename):
        # Save uploaded file
        temp_path = 'static/uploads/temp_upload.jpg'
        file.save(temp_path)

        # Desaturate input image
        img = cv2.imread(temp_path)
        img = desaturate(img)
        cv2.imwrite('static/uploads/temp_upload.jpg', img)
        flag = 0

        try:
            # Detect objects in image
            model = YOLO('models/yolo11n.pt')
            results = model.predict(temp_path)

            # Choose appropriate models based on content
            if 0 in results[0].boxes.cls:  # If person detected
                try:
                    # Process portrait image
                    model_path1 = 'models/finetuned_unsplash_1500_epoch8.keras'
                    model_path2 = 'models/pascal_voc_colorize_model_epoch_10.keras'

                    img_path = temp_path
                    img = cv2.imread(img_path)
                    img_resized = cv2.resize(img, (512, 512))
                    cv2.imwrite('static/temp/resized.jpg', img_resized)

                    # Colorize resized image
                    cv2.imwrite('static/temp/temp_colorized.jpg',
                                colorize_image('static/temp/resized.jpg', model_path1, model_path2))
                    img1 = cv2.imread('static/temp/temp_colorized.jpg')
                    print("Resizing image")
                    img1 = cv2.resize(img1, (512, 512),
                                      interpolation=cv2.INTER_CUBIC)
                    print("Resized image")

                    # Enhance colors
                    img1 = blend_images(img_resized, img1)
                    img1 = increase_saturation(img1, scale=1.6)
                    img1 = increase_vibrance(img1, strength=0.6)

                    # Process individual people in the image
                    l = crop_and_colorize(
                        'static/temp/resized.jpg', model_path1, model_path2)

                    # Apply masks and blend colorized portions
                    for image_path, bbox in l:
                        image = cv2.imread(image_path)
                        mask1 = mask(image_path)
                        mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
                        image[:, :, 3] = mask1

                        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)
                        roi = img1[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                        # Blend using alpha channel
                        alpha = image[:, :, 3] / 255.0
                        inv_alpha = 1 - alpha
                        for c in range(3):
                            roi[:, :, c] = (inv_alpha * roi[:, :, c] +
                                            alpha * image[:, :, c]).astype(np.uint8)
                        roi[:, :, 3] = (255 * np.maximum(alpha,
                                                         roi[:, :, 3] / 255)).astype(np.uint8)

                        img1[bbox[1]:bbox[3], bbox[0]:bbox[2]] = roi
                        img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)

                    # Final processing
                    img1 = cv2.resize(
                        img1, (img.shape[0], img.shape[1]), cv2.INTER_CUBIC)
                    img1 = blend_images(img, img1)
                    img1 = increase_saturation(img1, scale=1.6)
                    img1 = increase_vibrance(img1, strength=0.6)
                    cv2.imwrite('static/final.jpg', img1)

                except Exception as e:
                    # Fallback processing for portraits
                    print(e)
                    model_path1 = 'models/finetuned_unsplash_1500_epoch8.keras'
                    model_path2 = 'models/pascal_voc_colorize_model_epoch_10.keras'
                    img5 = colorize_image(temp_path, model_path1, model_path2)
                    img12 = cv2.imread(temp_path)
                    img11 = cv2.resize(img5, (img12.shape[0], img12.shape[1]))
                    img11 = blend_images(img12, img11)
                    img11 = increase_saturation(img11, scale=1.6)
                    img11 = increase_vibrance(img11, strength=0.6)
                    cv2.imwrite('static/output.jpg', img11)
                    flag = 1
            else:
                # Process landscape/non-portrait image
                model_path1 = 'models/finetuned_landscape5.keras'
                model_path2 = 'models/pascal_voc_colorize_model_epoch_10.keras'
                colorize_image(temp_path, model_path1, model_path2)

            # Determine output filename based on processing path
            if 0 in results[0].boxes.cls:
                output_filename = 'final.jpg'
            else:
                output_filename = 'output.jpg'

            if flag == 1:
                output_filename = 'output.jpg'

            # Cleanup and render result
            os.remove(temp_path)
            cleanup_temp_files()
            return render_template('index.html', colorized_image=output_filename)

        except Exception as e:
            # Handle errors
            if os.path.exists(temp_path):
                os.remove(temp_path)
            cleanup_temp_files()
            return f"Error processing image: {str(e)}", 500
    else:
        return "Invalid file type. Please upload a JPG or PNG image.", 400


if __name__ == "__main__":
    # app.run(debug=True)
    port = int(os.environ.get("PORT", 5000))  # Render dynamically assigns a port
    app.run(host="0.0.0.0", port=port, debug=True)  # Bind to 0.0.0.0
