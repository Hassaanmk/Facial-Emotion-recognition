

# **Facial Emotion Recognition** 😃😢😡  

This repository contains **Facial Emotion Recognition (FER)** models trained on the **FER** and **AffectNet** datasets using **Convolutional Neural Networks (CNNs)**. Additionally, it includes experiments with a **pretrained Vision Transformer (ViT)** for facial expression recognition. Real-time emotion detection scripts using a webcam are also provided.  


 **CNN-based Emotion Recognition**  
- Trained on **FER** and **AffectNet** datasets.  
- Evaluated prediction accuracy for each dataset.  
- Training scripts and datasets included.  

**ViT Transformer-based Recognition**  
- Used the **pretrained ViT model** from Hugging Face:  
  [ViT-Facial-Expression-Recognition](https://huggingface.co/motheecreator/vit-Facial-Expression-Recognition)  
- Evaluated performance on **FER** and **AffectNet** datasets.  

 ## Real-time Facial Expression Detection  
- Implemented real-time webcam-based emotion recognition.  
- Uses **OpenCV** and a trained model for live inference.  


## Getting Started  

###  1. Clone the Repository  
```bash
git clone https://github.com/Hassaanmk/Facial-Emotion-recognition.git
cd Facial-Emotion-recognition
```

###  2. Install Dependencies  
Ensure you have Python and the required libraries installed:  
```bash
pip install -r requirements.txt
```  

###  3. Training CNN Models  
Run training scripts inside the **FER training** or **AffectNet training** folders:  


###  4. Running ViT Model on FER and AffectNet 
Inside the `VIT Transformer` folder, run:  
```bash
python vit_evaluate.py
```

###  5. Real-time Facial Expression Detection
Run the webcam-based detection script:  
```bash
python webcam_test/webcam_detect.py
```



## Model Performance  

| Model | Dataset | Accuracy |
|--------|---------|----------|
| **CNN** | FER | 60.37% |
| **CNN** | AffectNet | 66% |
| **ViT Transformer** | FER | 87.0% |
| **ViT Transformer** | AffectNet | 57.0% |

  



## Datasets Used

- **[FER Dataset](https://www.kaggle.com/datasets/msambare/fer2013)**  
- **[AffectNet Dataset](https://www.researchgate.net/publication/315159311_AffectNet_A_Database_for_Facial_Expression_Valence_and_Arousal_Computing_in_the_Wild)**  



## References & Acknowledgments

- Hugging Face Pretrained ViT Model: [ViT-Facial-Expression-Recognition](https://huggingface.co/motheecreator/vit-Facial-Expression-Recognition)  
- OpenCV for real-time webcam-based inference  
- PyTorch for model training  



## Future Improvements 

- Implement **attention-based mechanisms** to improve ViT accuracy.  
- Experiment with **ResNet** and **EfficientNet** architectures.  
- Deploy as a **web application** or **mobile app**.  




