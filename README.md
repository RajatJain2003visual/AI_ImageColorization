# AI Image Colorizer 🎨

![](static/assets/Sequence_01.gif)

A sophisticated web application that transforms black and white images into vibrant, colorized versions using advanced deep learning techniques. The application intelligently detects image content and applies specialized colorization models for portraits and landscapes.

## Features ✨

- Automatic content-aware colorization
- Specialized processing for portraits and landscapes
- Real-time processing feedback with animated brush effect
- Support for multiple image formats (JPG, PNG, JPEG)
- Advanced color enhancement and blending
- User-friendly interface with drag-and-drop support
- Instant download of colorized images

## Screenshots 📸

### Home Interface
![Home Interface](static/assets/ss1.png)
![](static/assets/ss2.png)
*Clean, intuitive interface with drag-and-drop functionality*

### Processing Animation
![Processing Animation](static/assets/ss3.png)


### Portrait Colorization
![Portrait Result](static/assets/ss4.png)
*Example of portrait colorization with enhanced facial details*

### Landscape Colorization
<!-- ![Landscape Result](static\assets\ss4.png) -->

<div style="display: flex; gap: 10px;">
    <img src="static/assets/landscape_de.jpg" alt="Black and White Landscape" width="45%">
    <img src="static\assets\output (7).jpg" alt="Colorized Landscape" width="45%">
</div>
*Example of landscape colorization with natural color reproduction* 


## Technology Stack 🛠

- **Backend**: Flask, Python
- **Image Processing**: OpenCV, NumPy
- **AI Models**: 
  - YOLO for object detection
  - Custom Keras models for colorization
  - MediaPipe for segmentation
- **Frontend**: HTML5, CSS3, JavaScript

## Datasets Used in the project
- **Base Dataset**: pascal-voc-2012-dataset
- **Portrait Dataset**: CelebA-HQ, FFHQ, Unsplash
- **Landscape Dataset**: https://huggingface.co/datasets/mertcobanov/nature-dataset
- **Unsplash Images**

# How Image is colorizing
![How image is colorizing](static/assets/Model-Architecture.jpg)
 
## Installation 💻

1. Clone the repository:
```bash
git clone https://github.com/RajatJain2003visual/AI_ImageColorization.git
cd AI_ImageColorization
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/Scripts/activate ```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models to `models` directory:
- yolo11n.pt
- yolo11n-pose.pt
- finetuned_unsplash_1500_epoch8.keras
- finetuned_landscape5.keras
- pascal_voc_colorize_model_epoch_10.keras
- selfie_multiclass_256x256.tflite

5. Create required directories:
```bash
mkdir -p static/temp static/uploads
```

6. Run the application:
```bash
python app.py
```

7. Open `http://localhost:5000` in your browser

## Usage 📝

1. Open the website in your browser
2. Drop an image or click to select one
3. Wait for the colorization process to complete
4. Download your colorized image

## Project Structure 📁

```
ai-image-colorizer/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── yolo11n.pt
│   ├── yolo11n-pose.pt
│   └── ...
├── static/
│   ├── temp/
│   └── uploads/
└── templates/
    └── index.html
```

## Technical Details 🔧

- Uses LAB color space for natural color blending
- Implements adaptive saturation and vibrance enhancement
- Features intelligent portrait detection and segmentation
- Employs multiple AI models for optimal results
- Includes error handling and cleanup procedures

## Performance Optimization 🚀

- Efficient image processing pipeline
- Optimized model loading and inference
- Automatic cleanup of temporary files
- Responsive design for all screen sizes

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License

## Acknowledgments 🙏

- YOLO for object detection
- MediaPipe for segmentation
- Flask community for web framework
- OpenCV for image processing

### Contact & Project Links

[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:rajatofficial5940@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/RajatJain2003visual/AI_ImageColorization)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rajat-jain-29a04b236/)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Jpgxu8jFPNYUSxVSQn2-XK3utF231r9d?usp=sharing)


---
