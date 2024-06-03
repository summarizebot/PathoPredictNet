# PathoPredictNet
PathoPredictNet is a biomedical vision-language foundation model that has been pretrained on a comprehensive dataset of multimodal biomedical information. It employs advanced contrastive learning techniques to refine its understanding of complex medical data. The model integrates a specialized text encoder and a state-of-the-art image encoder, both optimized for the biomedical domain. Capable of performing a variety of vision-language processing tasks, PathoPredictNet excels in cross-modal retrieval, precise image classification, and responsive visual question answering. 


# Model
[Download Model](https://drive.google.com/file/d/1yNCS9FrB8EP9-7gbTY6FAVYS8SRU9JAJ/view?usp=sharing)

# Usage

import pydicom
import torch
from transformers import CLIPProcessor, CLIPModel

model_name = "pathopredict_model"  # Replace with the actual model name
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

def load_dicom_image(file_path):
    """
    Load a DICOM image and convert it to a format suitable for the CLIP model.
    """
    dicom = pydicom.dcmread(file_path)
    image = dicom.pixel_array
    # Normalize the image
    image = (image - image.min()) / (image.max() - image.min())
    # Convert single-channel to three channels if necessary
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    return image

def vqa_with_biomedclip(dicom_image_path, question):
    """
    Perform Visual Question Answering on a DICOM image using the BiomedCLIP model.
    """
    # Load and preprocess the image
    image = load_dicom_image(dicom_image_path)
    
    # Encode the question and image
    inputs = processor(text=question, images=image, return_tensors="pt", padding=True)
    
    # Perform inference
    outputs = model(**inputs)
    
    # The model outputs logits for matching between the image and the text
    logits_per_image = outputs.logits_per_image
    logits_per_text = outputs.logits_per_text
    
    # Extract the most relevant information (assuming VQA task requires the text logits)
    answer_index = logits_per_text.argmax()
    answer = question[answer_index]
    
    return answer

dicom_image_path = "path_to_dicom_image.dcm"  # Replace with your DICOM image path
question = "What abnormality is seen in this image?"

answer = vqa_with_biomedclip(dicom_image_path, question)
print(f"Answer: {answer}")
