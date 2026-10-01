# CNN Image Classification

A deep learning image classifier with an interactive web interface.

I built two convolutional neural networks from scratch, benchmarked them against ResNet50 and EfficientNetB2 on the CIFAR-10 dataset, and served predictions through a small Flask app where you can upload a photo and see what each model thinks it is.

## What It Does

* Trains custom CNNs on CIFAR-10 (60,000 small images across 10 classes like airplane, bird, cat and dog)
* Compares 3x3 filters against 5x5 filters to see how filter size affects accuracy and speed
* Visualizes the filters the first layer learns
* Tests transfer learning with ResNet50 and EfficientNetB2 backbones
* Classifies real photos and shows the top 3 guesses with confidence scores

## Results

| Model | Parameters | Test Accuracy | Time per Epoch |
|---|---|---|---|
| Custom CNN (3x3) | 1.08M | **72.1%** | 7.6s |
| Custom CNN (5x5) | 1.13M | 68.7% | 10.1s |
| ResNet50 (frozen) | 23.9M | 59.4% | 180.3s |
| EfficientNetB2 (frozen) | 8.0M | 10.0% | 129.1s |

**What I learned**

* The small 3x3 network beat every other model. Stacking small filters captures detail well on tiny 32x32 images and trains faster.
* Larger 5x5 filters added parameters but overfit more and scored lower.
* Frozen pretrained backbones struggled because they were trained on large 224x224 photos, not tiny CIFAR images.
* EfficientNetB2 stayed at chance level (10%), which points to an input preprocessing mismatch. Fixing that and fine tuning the backbone is my next step.

## Custom CNN Architecture

```
Input 32x32x3
Conv 32 → BatchNorm → ReLU
Conv 32 → BatchNorm → ReLU → MaxPool
Conv 64 → BatchNorm → ReLU → MaxPool
Flatten → Dense 256 → Dropout 0.5 → Dense 10 (Softmax)
```

Trained with Adam (learning rate 0.001), batch size 64, for 20 epochs.

## Project Structure

```
cnn-image-classification/
├── cnn_image_classification.ipynb   # Full training and evaluation notebook
├── flask_app/                       # Web interface for predictions
├── models/                          # Trained models (.keras and .h5)
├── results/                         # Training curves and filter plots
├── testImages/                      # Sample photos (airplane, bird, dog)
└── README.md
```

## How to Run

**Train the models (Google Colab)**

1. Open [colab.research.google.com](https://colab.research.google.com) and upload `cnn_image_classification.ipynb`
2. Go to `Runtime`, then `Change runtime type`, and pick `T4 GPU`
3. Click `Run all`. Training takes about 45 minutes and saves every model automatically.

**Classify your own photos (Flask)**

```bash
cd flask_app
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000` in your browser and upload an image.

## Tech Stack

Python, TensorFlow and Keras, NumPy, Matplotlib, Flask, Google Colab

## Author

Anh Tran
