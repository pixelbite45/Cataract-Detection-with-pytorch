# Cataract-Detection-with-pytorch
A Flask web app that uses a CNN deep learning model to detect cataracts from eye images. Users can upload an image and get real-time predictions with confidence scores. Built with PyTorch for modeling and Bootstrap for a modern, responsive interface, making cataract detection simple and accessible.

This project is a deep learning-powered Flask web application that detects **cataracts** from eye images using a **Convolutional Neural Network (CNN)**. Users can upload an eye image and receive real-time predictions with confidence scores.

---
![Cataract Detection](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Cataract_in_human_eye.jpg/640px-Cataract_in_human_eye.jpg)


## Features

- Upload eye images via a simple, clean web interface  
- Predicts whether the eye has cataract or is normal  
- Displays prediction confidence score  
- Responsive and mobile-friendly UI using Bootstrap 5  

---

## Technologies Used

- Python & Flask (Web framework)  
- PyTorch (Deep learning model)  
- Bootstrap 5 (Frontend styling)  
- OpenCV & Pillow (Image processing)  

---

<pre><code>## 📂 Dataset Structure ``` processed_image/ ├── train/ │ ├── cataract/ │ └── normal/ ├── test/ │ ├── cataract/ │ └── normal/ ``` </code></pre>

The CNN model is trained on this dataset to classify eye images as `cataract` or `normal`.


