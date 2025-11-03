# HAND SIGN RECOGNITION USING AI

## Objective
learns from labeled hand sign images and predicts what sign the hand is shwoing

## How it works

### Step 1 : Image Processing
>> each iamge is gray scaled
>> resized to **64x64** pixels for faster training
>> the pixel values are scaled between 0 to 1

### Step 2 : Learning
>> AI uses **Convolutional Neural Network (CNN)**
>> CNN learns **patterns and edges** in the hand
>> it figures out which features match each hand sign

### step 3 : Prediction
>> The trained model takes a new image
>> It compares the image with what it learned
>> It gives a label (like 0–5) as the final output

## Dataset
>> From **Kaggle** (`shivamaggarwal513/dlai-hand-signs-05`)
>> Images are already grouped by gesture type
>> Example labels: 0, 1, 2, 3, 4, 5

## 4. Evaluation
>> **Accuracy:** How many signs were correctly predicted
>> **Confusion Matrix:** Shows which signs get mixed up
>> **Loss Curve:** Shows how the model improves during training

## 5. Uses
>> Sign language recognition  
>> Gesture control for devices  
>> AR/VR or gaming interfaces  
>> Hands-free robot control

## 6. Limits
>> Struggles with poor lighting or weird backgrounds.  
>> Needs clear, front-facing hand photos.  
>> Real-time use needs faster hardware or optimization.

## 7. Conclusion
> A simple CNN can recognize hand signs fast and accurately using grayscale images.  
> This shows how AI can understand human gestures and make tech more natural to use.

---

# **Keywords:** AI, CNN, Hand Signs, Image Recognition, Computer Vision
