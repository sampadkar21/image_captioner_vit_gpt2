# 🖼️ Image Captioning using Vision Transformer (ViT) and GPT2

This project demonstrates an **end-to-end image captioning pipeline** using a **Vision Transformer (ViT)** as the image encoder and **GPT-2** as the language decoder, trained on the **Flickr30K** dataset.

---

## 🚀 Overview

This deep learning model takes an image as input and generates a descriptive caption in natural language. It uses a transformer-based encoder-decoder architecture implemented using Hugging Face Transformers and PyTorch.

- **Encoder:** ViT (`google/vit-base-patch16-224-in21k`)
- **Decoder:** GPT2 with cross-attention enabled
- **Dataset:** [Flickr30K](https://www.kaggle.com/datasets/adityajn105/flickr30k-images)
- **Metric:** BLEU Score

---

## 📁 Dataset

- **Source:** [Kaggle - Flickr30K](https://www.kaggle.com/datasets/adityajn105/flickr30k-images)
- **Images:** 30,000
- **Captions:** 5 per image (but only one used per image in this example)
- **Preprocessing:**
  - Removed special characters from captions
  - Converted all text to lowercase
  - Filtered null entries

---

## 🧠 Model Architecture

- **ViT Encoder:** (https://huggingface.co/docs/transformers/model_doc/vit)
  - Pretrained on ImageNet21K
  - Parameters frozen (non-trainable)
  - ![img](https://production-media.paperswithcode.com/methods/Screen_Shot_2021-01-26_at_9.43.31_PM_uI4jjMq.png)

- **GPT2 Decoder:** (https://huggingface.co/openai-community/gpt2)
  - `gpt2` with `add_cross_attention=True`
  - `<|pad|>` token added for batching
  - Trained with causal language modeling loss
  - ![img](https://bea.stollnitz.com/images/gpt-transformer/3-transformer.png)

---

## 🏋️‍♂️ Training Details

- **Loss Function:** Causal LM Loss
- **Optimizer:** AdamW (lr = 1e-4, weight decay = 0.0005)
- **Batch Size:** 32
- **Epochs:** 1
- **Train-Validation Split:** 95%-5%
- **Cross-validation:** ❌ Not used
- **Gradient Clipping:** Enabled (max norm = 1.0)

---

## 📊 Results

- **Epochs Trained:** 1
- **Final Training Loss:** `0.336`
- **Validation BLEU Score:** `90.36` (on first epoch)

> ⚠️ Note: These results were achieved using a single caption per image and only 1 epoch. Full potential may require more epochs and data augmentations.

---

## 🖼️ Inference Example

You can generate captions for any image URL using:

```python
test_model(trained_model, image_url, tokenizer, processor, device)
```
![image](https://github.com/user-attachments/assets/608e01d8-9a6a-4a75-96ca-683d29b85e87)
![image](https://github.com/user-attachments/assets/dc6e0fd4-6027-4bf8-9c2f-3668f254b32a)

