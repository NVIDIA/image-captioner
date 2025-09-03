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

import os
import base64
import requests
import gradio as gr
import tempfile
from huggingface_hub import InferenceClient


def resize_image(img_path, max_size):
    """
    Fast image resizing using OpenCV
    :param img_path: path to image
    :param max_size: maximum size supported - the larger side of the image will be set to this
    :return: a temporary path with the resized image
    """
    try:
        import cv2
        
        img = cv2.imread(img_path)
        if img is None:
            raise gr.Error(f"Could not read image: {os.path.basename(img_path)}")
        
        height, width = img.shape[:2]
        if max(width, height) <= max_size:
            return img_path
        
        if width > height:
            new_width = max_size
            new_height = int(max_size * height / width)
        else:
            new_height = max_size
            new_width = int(max_size * width / height)
        
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        temp_fd, temp_img_path = tempfile.mkstemp(suffix='.jpg', prefix='resized_')
        os.close(temp_fd)
        cv2.imwrite(temp_img_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        return temp_img_path
        
    except ImportError:
        raise gr.Error("OpenCV is required for image processing. Install with: pip install opencv-python")
    except Exception as e:
        if 'temp_img_path' in locals() and os.path.exists(temp_img_path):
            try:
                os.remove(temp_img_path)
            except:
                pass
        raise gr.Error(f"Error resizing image '{os.path.basename(img_path)}': {str(e)}")


def hugging_face_models(model_name, img_path, api_key, max_tokens):
    """
    Generates a custom tag using the specified model from Hugging Face.
    Base code from - https://huggingface.co/Salesforce/blip-image-captioning-base?inference_api=true
    :param model_url: desired model's HF url 
    :param img_path: absolute path to image
    :param api_key: API key/token required to make calls to the model
    :return: returns the generated tag text
    """
    client = InferenceClient(
        provider="auto",
        api_key=api_key,
    )

    with open(img_path, "rb") as f:
        data = base64.b64encode(f.read()).decode('utf-8')
        
    try:
        response = client.chat.completions.create(
            model= model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Describe this image in {max_tokens} tokens or less. Do not include any other text in your response."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{data}"
                            }
                        }
                    ]
                }
            ],
        )
        
        if not response:
            raise gr.Error("There was an error in fetching the response. "
                       "Please check if your API key is valid and has the required access")

        return response.choices[0].message.content
    
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        if status_code in [404, 503]:
            raise gr.Error(f"Model unavailable (status code: {status_code}). The model is either not hosted or temporarily unavailable on Hugging Face.")
        else:
            raise gr.Error(f"HTTP Error: {e}")
    except (requests.exceptions.ConnectionError, requests.exceptions.RequestException) as e:
        raise gr.Error(f"Connection error: Unable to reach Hugging Face API. {str(e)}")
    except Exception as e:
        raise gr.Error(f"Error: {str(e)}")


def nvidia_ai_models(model_url, img_path, api_key, max_tokens, temperature, top_p, model_name=None, chat=False):
    """
    Generates a custom tag using the chosen NVIDIA NIM model.
    Base code from - https://build.nvidia.com/microsoft/microsoft-kosmos-2
    :param model_url: URL endpoint for the NVIDIA NIM model
    :param img_path: absolute path to image
    :param api_key: API key/token required to make calls to the model
    :param max_tokens: The maximum number of tokens to generate
    :param temperature: Temperature value used for text generation
    :param top_p: top_p sampling mass
    :param model_name: Name of the model (used when chat=True)
    :param chat: Boolean flag to enable chat mode
    :return: returns the generated tag text
    """
    img_size = os.path.getsize(img_path)
    if img_size > 180_000:
        img_path = resize_image(img_path, 180_000)

    with open(img_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    # Clean up temporary resized file
    if os.path.exists(img_path) and "resized_" in os.path.basename(img_path):
        try:
            os.remove(img_path)
        except:
            pass

    headers = {
      "Authorization": f"Bearer {api_key}",
      "Accept": "application/json"
    }

    instruction = (
        f"Write a single concise caption in <= {max_tokens} tokens. Output only the caption."
    ) if max_tokens <= 25 else (
        f"Write a detailed description in around {max_tokens} covering subjects, key attributes, spatial relations, actions, and scene context. Avoid redundancy and output only the description."
    )

    payload = {
        "messages": [
          {
            "role": "user",
            "content": f"{instruction}\n\nImage: <img src=\"data:image/png;base64,{image_b64}\" />"
          }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
      }
    
    if chat == True:
      payload["model"] = model_name
      payload["stream"] = False

    response = requests.post(model_url, headers=headers, json=payload)

    if not response:
        raise gr.Error("There was an error in fetching the response. "
                       "Please check if your API key is valid and has the required access, and whether advanced paramters (if any) are valid")

    return response.json()["choices"][0]['message']['content']