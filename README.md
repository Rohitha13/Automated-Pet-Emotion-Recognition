# Pet Facial Emotion Recognition 🐶🐱

A deep learning project that looks at a picture of a pet (dog or cat) and guesses how it's feeling — **happy**, **sad**, **angry**, or **other**.

---

## What this project does

You give it a photo of a pet. It does two things:

1. **Finds the pet in the picture** (using YOLOv8 to draw a box around the face/body).
2. **Guesses the emotion** on the pet's face (using a trained image classifier).

So the final output is something like: *"Dog detected — emotion: Happy 🙂"*

---

## What's in this folder

| File | What it is |
|------|------------|
| `pet_facial_emotion_Basline_Dense.ipynb` | **Model 1 — Baseline.** The simplest version. Just a plain neural network (Dense layers only). Used as a starting point to compare against. Accuracy: ~50% on test set. |
| `pet_facial_emotion_AdvanceCNN.ipynb` | **Model 2 — Advanced CNN with Transfer Learning.** Combines an *animal classifier* (cat vs. dog vs. wild) with an *emotion classifier* on top. Accuracy: ~62%. |
| `pet_facial_emotion_EfficientNet.ipynb` | **Model 3 — EfficientNet.** Uses a pre-trained EfficientNet model and fine-tunes it for emotions. Usually the best-performing one. |
| `efficinetb5.ipynb` | Experiment using the larger **EfficientNetB5** backbone. |
| `cnnprediction.ipynb` | Script to run predictions with the CNN emotion model on new images. |
| `predictenn.ipynb` | Script that combines **YOLOv8 (pet detection)** + **NN (emotion)** for the full pipeline on new images. |
| `animal_classifier_model.h5` | Pre-trained model that tells cats, dogs, and wild animals apart. Used inside the Advanced CNN. |
| `yolov8n.pt` | Pre-trained YOLOv8 model used to detect the pet in the image. |

---

## The emotions it can recognize

- 😊 **Happy**
- 😢 **Sad**
- 😠 **Angry**
- 🐾 **Other** (neutral / not clearly one of the above)

---

## How it works, step by step

```
  Input image
       │
       ▼
  ┌─────────────┐
  │   YOLOv8    │   ← finds the pet and crops its face
  └─────────────┘
       │
       ▼
  ┌─────────────────────┐
  │  Animal Classifier  │   ← is it a cat, dog, or wild?
  └─────────────────────┘
       │
       ▼
  ┌─────────────────────┐
  │  Emotion Classifier │   ← happy / sad / angry / other
  └─────────────────────┘
       │
       ▼
  Final prediction (with a box drawn around the pet)
```

---

## The three models, compared

| Model | Approach | Accuracy | Notes |
|-------|----------|----------|-------|
| **Baseline (Dense)** | Simple fully-connected network | ~50% | Very basic — here for comparison only |
| **Advanced CNN** | Custom CNN + transfer learning from animal classifier | ~62% | A big jump over the baseline |
| **EfficientNet** | Pre-trained EfficientNet fine-tuned on pet emotions | Best | Recommended for real use |

---

## What you need to run it

Python 3, plus these libraries:

```bash
pip install tensorflow keras opencv-python numpy matplotlib pillow scikit-learn ultralytics seaborn pandas
```

---

## How to use it

### Option 1 — Just run a prediction on your own photo

Open `predictenn.ipynb`, change the image path to point to your photo, and run all cells. It will:
- Detect the pet
- Crop the face
- Predict the emotion
- Show the image with a labeled bounding box

### Option 2 — Train the models yourself

1. Prepare your dataset in this folder layout:
   ```
   pets_facial_expression_dataset/
   ├── happy/
   ├── Sad/
   ├── Angry/
   └── Other/
   ```
2. Open any of the training notebooks (`pet_facial_emotion_*.ipynb`)
3. Update the folder paths at the top to match yours
4. Run the cells in order

---

## Dataset

## Dataset

The models were trained on the **Pets Facial Expression Dataset** from Kaggle, which contains images of dogs and cats labeled with one of the four emotions above.

🔗 **Download it here:** [Pets Facial Expression Dataset on Kaggle](https://www.kaggle.com/datasets/anshtanwar/pets-facial-expression-dataset)

The animal classifier was trained separately on a 10,00-image animal-faces dataset (cat / dog / wild).


---

## Things to keep in mind

- Emotion recognition in pets is hard — even humans disagree on what a dog's face means. An accuracy of ~60–70% is actually pretty decent for this task.
- The model works best on clear, front-facing photos where the pet's face is visible.
- Weird angles, blurry photos, or multiple pets in one image may confuse it.

---

## Credits

- **YOLOv8** by Ultralytics — for pet detection
- **EfficientNet** (Google) — for the transfer-learning backbone
- Built with **TensorFlow / Keras**
