# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gradio as gr
import sys
import os

dam_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dam')
sys.path.append(dam_path)

import numpy as np
import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import cv2
from dam import DescribeAnythingModel, disable_torch_init
from scipy import ndimage
import requests
import base64
import io

MODEL_PATH = "nvidia/DAM-3B"

sam_model = None
sam_processor = None
dam = None
device = None

prompt_modes = {
    "focal_prompt": "full+focal_crop",
}

def extract_points_from_mask(mask_pil):
    mask = np.asarray(mask_pil)[..., 0]
    coords = np.nonzero(mask)

    # coords is in [(x, y), ...] format
    coords = np.stack((coords[1], coords[0]), axis=1)
    return coords

def add_contour(img, mask, color=(1., 1., 1.)):
    img = img.copy()

    mask = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness=6)

    return img

def get_mask_components(mask_np):
    """Split mask into separate connected components."""
    # Label connected components
    labeled_array, num_features = ndimage.label(mask_np)
    
    # Separate mask for each component
    component_masks = []
    for i in range(1, num_features + 1):
        component_mask = (labeled_array == i).astype(np.uint8)
        component_masks.append(component_mask)
    
    return component_masks

def encode_image_for_api(image):
    """Encode an image for the OpenAI API."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
    }

def combine_descriptions_with_llm(descriptions, api_key, model_choice, image=None):
    """Combine multiple region descriptions into a caption using the selected model API with image."""
    if not descriptions:
        return "No descriptions available to combine."
    
    if not api_key or api_key.strip() == "":
        return "Please provide an API key to generate captions."
    
    descriptions_text = "\n".join(descriptions)
    
    user_prompt = f"""Please weave these regional descriptions into a cohesive, flowing description:

{descriptions_text}

Instead of simply summarizing each region separately, create a natural narrative that connects these elements.
Use the image as context, and focus primarily on the user-selected regions as they represent what's important, while briefly setting context (for surrounding regions) based on the image.
Keep your response under 3 sentences while making sure the narrative flows naturally but is completely factual.

Avoid: naming specific people/animals, interpreting emotions, or adding fictional elements.
Do not use the word "region" in your response.
"""
    
    try:
        if model_choice == "GPT-4o (OpenAI)":
            # OpenAI GPT-4o configuration
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key.strip()}'
            }
            
            content = []
            content.append({"type": "text", "text": user_prompt})
            if image:
                content.append(encode_image_for_api(image))

            payload = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": content}
                ],
                "temperature": 0.7,
                "top_p": 0.7,
                "n": 1,
                "stream": False,
                "max_tokens": 1024,
                "presence_penalty": 0,
                "frequency_penalty": 0
            }
            
            url = 'https://api.openai.com/v1/chat/completions'
            
        elif model_choice == "VILA (NVIDIA)":
            # NVIDIA VILA configuration
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key.strip()}'
            }
            
            # Prepare content with image if available
            content = []
            content.append({"type": "text", "text": user_prompt})
            if image:
                content.append(encode_image_for_api(image))
            
            payload = {
                "model": "nvidia/vila",
                "messages": [
                    {"role": "system", "content": "detailed thinking off"},
                    {"role": "user", "content": content}
                ],
                "temperature": 0,
                "top_p": 0.95,
                "max_tokens": 4096,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "stream": False
            }
            
            url = 'https://ai.api.nvidia.com/v1/vlm/nvidia/vila'
            
        else:
            return f"Unsupported model choice: {model_choice}"
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            print(f"API request failed with status code: {response.status_code}")
            print(response.text)
            return f"Error generating combined caption: API returned status code {response.status_code}. Please check your API key and try again."
            
    except Exception as e:
        print(f"Error calling API: {e}")
        return f"Error generating combined caption: {str(e)}"

def describe(image, query, image_path):
    # Check if an image is loaded
    if image is None or not isinstance(image, dict) or 'background' not in image:
        raise gr.Error("Please select a directory and load an image first before describing regions.")
    
    if image_path is not None and image_path == "":
        raise gr.Error("Please select a directory and load an image first before describing regions.")
    
    print(image.keys())
    
    image['image'] = image['background'].convert('RGB')
    del image['background'], image['composite']
    assert len(image['layers']) == 1, f"Expected 1 layer, got {len(image['layers'])}"

    full_mask_np = (np.asarray(image['layers'][0])[..., 3] > 0).astype(np.uint8)
    
    component_masks = get_mask_components(full_mask_np)
    
    if len(component_masks) == 0:
        raise gr.Error("No regions selected")
    
    print(f"Found {len(component_masks)} separate regions to describe")
    
    img_np = np.asarray(image['image']).astype(float) / 255.
    combined_img_with_contour = img_np.copy()
    
    color_rgb = (1.0, 1.0, 1.0)
    
    all_descriptions = []
    for i, component_mask_np in enumerate(component_masks):
        print(f"Processing region {i+1}/{len(component_masks)}")
        
        component_mask_pil = Image.fromarray(component_mask_np * 255).convert('RGB')
        
        points = extract_points_from_mask(component_mask_pil)
        
        np.random.seed(0)
        
        if points.shape[0] == 0:
            print(f"No points in region {i+1}, skipping")
            continue
        
        # Randomly sample 8 points from the mask
        points_selected_indices = np.random.choice(points.shape[0], size=min(points.shape[0], 8), replace=False)
        points = points[points_selected_indices]
        
        print(f"Selected points for region {i+1} (to SAM): {points}")
        
        coords = [points.tolist()]
        
        refined_mask_np = apply_sam(image['image'], coords)
        component_mask = Image.fromarray(refined_mask_np)
        
        # Add contour to the combined image
        combined_img_with_contour = add_contour(combined_img_with_contour, refined_mask_np, color=color_rgb)
        
        # Get description for this component
        description_generator = dam.get_description(
            image['image'], 
            component_mask, 
            query, 
            streaming=True, 
            temperature=0.2, 
            top_p=0.5, 
            num_beams=1, 
            max_new_tokens=512, 
        )
        
        component_text = ""
        for token in description_generator:
            component_text += token
        
        # Add to the list of descriptions
        region_label = f"Region {i+1}: "
        all_descriptions.append(f"{region_label}{component_text}")
    
    # Combining all descriptions
    combined_img_with_contour_pil = Image.fromarray((combined_img_with_contour * 255.).astype(np.uint8))
    
    if not all_descriptions:
        text = "No valid regions found to describe."
    else:
        text = "\n\n".join(all_descriptions)
    
    return combined_img_with_contour_pil, text, all_descriptions, combined_img_with_contour_pil

def generate_caption(api_key, model_choice, region_descriptions, image=None):
    """Generate a caption using the selected model API with image."""
    if not region_descriptions:
        return "No descriptions available to combine."
    
    descriptions = region_descriptions
        
    return combine_descriptions_with_llm(descriptions, api_key, model_choice, image)

def apply_sam(image, input_points):
    global sam_model, sam_processor, device
    
    inputs = sam_processor(image, input_points=input_points, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = sam_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0][0]
    scores = outputs.iou_scores[0, 0]

    mask_selection_index = scores.argmax()

    mask_np = masks[mask_selection_index].numpy()

    return mask_np
