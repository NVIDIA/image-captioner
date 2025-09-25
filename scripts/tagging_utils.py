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
import ast
import json
import shutil
from typing import Dict, List, Tuple, Optional, Any, Union

import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from wordcloud import WordCloud, STOPWORDS

from model_call import nvidia_ai_models, hugging_face_models
from visualization import create_clusters_and_plot
from PIL import Image

# Supported models and their hosting services
model_host_mapping = {
    "Kosmos-2": "NVIDIA NIM",
    "Gemma 3 27B": "Hugging Face",  
    "Llama-3.2 90B Vision Instruct": "NVIDIA NIM", 
    "NVIDIA Vila": "NVIDIA NIM",
    "Llama Nemotron Nano VL": "NVIDIA NIM"
}

# Global Variables
curr_chosen_directory: Optional[str] = None  # Path to the selected directory
dir_images: Optional['ImagesInDir'] = None  # A subset of the files which are images in the supported formats
image_tag_clusters: Optional[pd.DataFrame] = None  # Dataframe with image-tag-cluster sets
number_of_clusters: Optional[int] = None  # Number of clusters entered by the user
curr_tag: int = 0  # Index of the current tag
curr_keyword: Optional[str] = None  # Current keyword to search for
filtered_df: Optional[pd.DataFrame] = None  # DataFrame of samples containing the keyword
viz_filtered_df: Optional[pd.DataFrame] = None  # DataFrame of samples to include in the filtered gallery


class ImagesInDir:
    """Manages a directory of images for tagging."""
    
    def __init__(self, directory_elements: List[str]):
        """Initialize with image file paths from the directory.
        Args:
            directory_elements: List of the chosen directory's components
        """
        self.image_paths = get_images_from_dir(directory_elements)
        self.idx = -1

    def next_image(self, next_img: bool) -> Tuple[str, str, str, str, str]:
        """Determine the next/previous image and return outputs required for the UI.
        Args:
            next_img: True for next image, False for previous image
        Returns:
            Tuple containing (image_path, image_path, tag, tag_path, generated_tag)
        """
        if next_img:
            self.idx += 1
            # Loop back to first image if past the end
            if self.idx >= len(self.image_paths):
                self.idx = 0
        else:
            self.idx -= 1
            # Move to last image if before the beginning
            if self.idx < 0:
                self.idx = len(self.image_paths) - 1

        cap_path = get_tag_path(self.image_paths[self.idx])

        # Read existing tag file or create a new one
        if os.path.exists(cap_path):
            tag = read_tag(cap_path)
        else:
            open(cap_path, 'w').close()
            tag = ""

        # Return image, image_path, tag, tag_path, generated tag (empty)
        return self.image_paths[self.idx], self.image_paths[self.idx], tag, cap_path, ""


# ----- Directory and Image Selection Functions -----

def select_directory(chosen_directory: List[str]) -> List[Any]:
    """Confirm chosen directory and populate directory images and elements.
    Args:
        chosen_directory: List of paths from FileExplorer component
    Returns:
        List containing image-tag information and directory path
    """
    global curr_chosen_directory, dir_images

    if not chosen_directory:
        raise gr.Error("Choose a directory above.")

    if curr_chosen_directory != chosen_directory[0]:
        dir_images = ImagesInDir(chosen_directory)
    curr_chosen_directory = chosen_directory[0]

    image_tag_info = dir_images.next_image(True)
    image_tag_pairs = [(path, read_tag(get_tag_path(path))) for path in dir_images.image_paths]
    
    return list(image_tag_info) + [curr_chosen_directory] + [image_tag_pairs]


def load_image(forward: bool) -> Tuple[str, str, str, str, str]:
    """Load the next/previous image-tag pair with required details.
    Args:
        forward: True for next image, False for previous
    Returns:
        Image-tag pair information
    """
    global dir_images, curr_chosen_directory

    if curr_chosen_directory is None:
        raise gr.Error("Please choose a directory first.")

    return dir_images.next_image(forward)


def show_next_image() -> Tuple[str, str, str, str, str]:
    """Load the next image in the UI."""
    return load_image(forward=True)


def show_prev_image() -> Tuple[str, str, str, str, str]:
    """Load the previous image in the UI."""
    return load_image(forward=False)


def get_tag_path(image_path):
    """
    Function to return the tag file's path for any given image.
    For image abc.jpg, the tag file will be abc.txt in the same directory

    :param image_path: the path to an image in the chosen directory
    :return: absolute path of the corresponding tag text file
    """
    image_file = os.path.basename(image_path)
    image_name = image_file.split(".")[0]
    root_path = os.path.dirname(image_path)
    c_path = f"{os.path.join(root_path, image_name)}.txt"
    return c_path


def get_images_from_dir(elements):
    """
    Function to return only image paths from the input directory.
    Three image formats are supported - png, jpg, jpeg
    :param elements: list of all files in the chosen folder - as given by the Gradio FileExplorer
    :return: list of image paths and their corresponding tag paths
    """
    image_paths = []
    valid_types = ("png", "jpg", "jpeg")

    for file in elements:
        if file.endswith(valid_types):
            image_paths.append(file)

    if not image_paths:
        raise gr.Error("Please make sure there are images in this directory.")

    return image_paths


def read_tag(path):
    """
    Reads and returns the text in given file
    :param path: absolute path to a tag (text) file
    :return: tag
    """

    if os.path.exists(path):
        f = open(path, "r")
        tag = f.read()
        f.close()

    else:
        tag = ""
    return tag


def get_params_from_string(input_string):
    """
    Returns a dictionary of parameters if given by the user
    :param input_string: string of comma separated parameters
    :return: dictionary in the format- {param1:value, param2:value...}
    """
    params = dict()
    param_strings = [p.strip() for p in input_string.split(",")]
    for p in param_strings:
        try:
            key, value = p.split("=")
        except Exception:
            raise gr.Error(f"No value given for parameter {p}.")
        params.update({key.strip(): float(value.strip())})

    return params


def get_tag(img_path):
    """
    Given an image path, loads the tag.
    :param img_path: path to chosen image
    :return: tag text and path (so that edits can be saved)
    """
    cap_path = get_tag_path(img_path)

    return read_tag(cap_path), cap_path






# ----- Tag Management Functions -----

def save_tag(tag: str, path: str) -> str:
    """Save tag text to the specified file path.
    Args:
        tag: Tag string (custom, edited, or generated)
        path: Tag file's absolute path
    Returns:
        The saved tag text
    """
    if not path:
        raise gr.Error("No tag path given.")

    with open(path, "w") as f:
        f.write(tag)
    
    return tag


def add_pre_and_suffix(prefix: str, suffix: str, curr_image_path: str) -> str:
    """Add prefix and/or suffix to all tags in the dataset.
    Args:
        prefix: String to add as prefix
        suffix: String to add as suffix
        curr_image_path: Path to image currently displayed
    Returns:
        Updated tag for the current image
    """
    global dir_images
    current_tag = None
    
    for img_path in dir_images.image_paths:
        tag, tag_path = get_tag(img_path)
        tag = f"{prefix} {tag} {suffix}"
        save_tag(tag, tag_path)

        if img_path == curr_image_path:
            current_tag = tag

    return current_tag


# ----- Model and Tag Generation Functions -----

def gen_tag_from_model(model: str, host: str, path: str, api_key: str, 
                       input_advanced_params: str, long_or_short: str, message: int = 0) -> str:
    """Generate a tag for an image using the specified model.
    Args:
        model: Model name
        host: Hosting service
        path: Path to the image
        api_key: Valid API key for the platform
        input_advanced_params: Comma-separated parameters
        long_or_short: "Long" or "Short" tag preference
        message: Whether to show info message (0=yes, 1=no)
    Returns:
        Generated tag text
    """
    if not path:
        raise gr.Error("No image given")

    # Parse advanced parameters if provided
    advanced_params = get_params_from_string(input_advanced_params) if input_advanced_params else None
    
    # Set max tokens based on tag length preference
    max_tokens = 100 if long_or_short == "Long" else 25

    if host == "NVIDIA NIM":
        if model == "Kosmos-2":
            model_url = "https://ai.api.nvidia.com/v1/vlm/microsoft/kosmos-2"
            model_name = None
            chat = False
        elif model == "Llama-3.2 90B Vision Instruct":
            model_url = "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct/chat/completions"
            chat = True
            model_name = 'meta/llama-3.2-90b-vision-instruct'
        elif model == "NVIDIA Vila":
            model_url = "https://ai.api.nvidia.com/v1/vlm/nvidia/vila"
            chat = True
            model_name = 'nvidia/vila'

        elif model == "Llama Nemotron Nano VL":
            model_url = "https://integrate.api.nvidia.com/v1/chat/completions"
            chat = True
            model_name = 'nvidia/llama-3.1-nemotron-nano-vl-8b-v1'

        # Show parameter info message
        if message == 0:
            gr.Info("The supported advanced parameters are:\n\nmax_tokens, temperature, top_p")
        
        # Default parameter values
        temperature = 0.20
        top_p = 0.20

        # Override with user-provided values if available
        if advanced_params:
            if "max_tokens" in advanced_params:
                max_tokens = advanced_params["max_tokens"]
            if "temperature" in advanced_params:
                temperature = advanced_params["temperature"]
            if "top_p" in advanced_params:
                top_p = advanced_params["top_p"]

        tag = nvidia_ai_models(
            model_url=model_url,
            img_path=path,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p, 
            model_name=model_name, 
            chat=chat
        )

    elif host == "Hugging Face":
        if model == "Gemma 3 27B":
            name = "google/gemma-3-27b-it"
        # Can add more HF models here

        if message == 0:
            gr.Info("There are no supported advanced parameters")
            
        tag = hugging_face_models(model_name=name, img_path=path, api_key=api_key, max_tokens=max_tokens)
    else:
        raise gr.Error("Please choose a model to continue")

    return tag


def gen_tag_all(model: str, host: str, api_key: str, curr_img_path: str, 
                advanced_params: str, long_or_short_tag: str) -> str:
    """Generate and save tags for all images in the directory.
    Args:
        model: Model name
        host: Hosting service
        api_key: User's API key
        curr_img_path: Path to currently open image
        advanced_params: Advanced parameters string
        long_or_short_tag: "Long" or "Short" tag preference
    Returns:
        Current image's generated tag
    """
    global dir_images
    current_tag = ""
    message = 0
    
    for img_path in dir_images.image_paths:
        tag = gen_tag_from_model(model, host, img_path, api_key, advanced_params, long_or_short_tag, message)
        cap_path = get_tag_path(img_path)
        save_tag(tag, cap_path)
        message = 1  # Only show info message for the first image

        if img_path == curr_img_path:
            current_tag = tag
            
    return current_tag


def update_host(model: str) -> Dict[str, str]:
    """Update the hosting service textbox based on the selected model.
    Args:
        model: Model chosen in the dropdown
    Returns:
        Gradio update object with the hosting service
    """
    if model in model_host_mapping:
        return gr.update(value=model_host_mapping[model])
    return gr.update(value="")


# ----- Clustering and Visualization Functions -----

def cluster_and_plot(data_directory: str, num_clusters: int, image_or_tag: str) -> Optional[plt.Figure]:
    """Cluster the dataset and return a 2D plot.
    Args:
        data_directory: Dataset directory path
        num_clusters: Number of clusters
        image_or_tag: "Images" or "Tags" to cluster by
    Returns:
        Matplotlib plot figure
    """
    global image_tag_clusters, number_of_clusters
    number_of_clusters = num_clusters
    plot = None
    
    if image_or_tag == "Images":
        plot, image_tag_clusters = create_clusters_and_plot(data_directory, num_clusters, False)
    elif image_or_tag == "Tags":
        plot, image_tag_clusters = create_clusters_and_plot(data_directory, num_clusters, True)

    return plot


def gen_wordcloud(cluster_number: int) -> Tuple[plt.Figure, pd.DataFrame]:
    """Generate a word cloud for a specific cluster's tags.
    Args:
        cluster_number: Cluster to visualize (0-based)
    Returns:
        Tuple of (word cloud plot, word frequency dataframe)
    """
    global image_tag_clusters, number_of_clusters

    if image_tag_clusters is None:
        raise gr.Error("Please cluster the samples first.")

    if cluster_number >= number_of_clusters or cluster_number < 0:
        raise gr.Error("Invalid cluster number")

    # Get tags from the specified cluster
    cluster_df = image_tag_clusters[image_tag_clusters['cluster'] == str(cluster_number)]
    text = " ".join(tag for tag in cluster_df.Tag)

    # Generate word cloud
    stopwords = set(STOPWORDS)
    plt.figure()
    wc = WordCloud(stopwords=stopwords, background_color="white", collocations=False).generate(text)
    plt.imshow(wc)
    plt.axis("off")

    # Create frequency dataframe
    freq = {}
    for word in text.split():
        freq[word] = freq.get(word, 0) + 1

    # Filter to only include words in the wordcloud
    freq = {k: v for k, v in freq.items() if k in wc.words_}
    df = pd.DataFrame({"Word": list(freq.keys()), "Frequency": list(freq.values())})
    
    return plt, df


def load_filtered_grid(cluster_number: Optional[int], keywords: Optional[str]) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Return image-tag pairs from a cluster or filtered by keywords.
    Args:
        cluster_number: Cluster to filter by (0-based)
        keywords: Comma-separated keywords to filter by
    Returns:
        Tuple of (image-tag pairs for gallery, list of image paths)
    """
    global image_tag_clusters, number_of_clusters, dir_images, viz_filtered_df

    if cluster_number is not None:
        if image_tag_clusters is None:
            raise gr.Error("Please cluster the samples first.")

        if cluster_number >= number_of_clusters or cluster_number < 0:
            raise gr.Error("Invalid cluster number")

        viz_filtered_df = image_tag_clusters[image_tag_clusters['cluster'] == str(cluster_number)]
        
        # Create image-tag pairs reading from file to get the latest tags
        grid = [
            (viz_filtered_df.iloc[i]['Image'], 
             f"{read_tag(get_tag_path(viz_filtered_df.iloc[i]['Image']))} \n {viz_filtered_df.iloc[i]['Image']}")
            for i in range(len(viz_filtered_df.Image))
        ]

    elif keywords is not None:
        if dir_images is None:
            raise gr.Error("Select a dataset in the Tag Dataset tab first.")

        # Create dataframe with all images and their tags
        viz_filtered_df = pd.DataFrame({
            'Image': list(dir_images.image_paths), 
            'Tag': [get_tag(i)[0] for i in list(dir_images.image_paths)]
        })
        
        # Filter by keywords (using OR operation between keywords)
        words = keywords.replace(",", "|")
        viz_filtered_df = viz_filtered_df[viz_filtered_df['Tag'].str.contains(words)]
        
        # Create image-tag pairs
        grid = [
            (viz_filtered_df.iloc[i]['Image'], 
             f"{read_tag(get_tag_path(viz_filtered_df.iloc[i]['Image']))} \n {viz_filtered_df.iloc[i]['Image']}")
            for i in range(len(viz_filtered_df.Image))
        ]

    else:
        raise gr.Error("Enter a cluster number OR keywords to filter the samples")

    return grid, viz_filtered_df.Image.tolist()


def display_filter_sample(selected_image: gr.SelectData) -> str:
    """Return the path of the selected image from the filtered gallery.
    Args:
        selected_image: Gradio SelectData event
    Returns:
        Path to the selected image
    """
    global viz_filtered_df
    index = selected_image.index
    return viz_filtered_df.iloc[index]['Image']


# ----- Find and Replace Functions -----

def find_text_in_captions(text: str, index: int) -> Tuple[str, str, List[Tuple[str, str]]]:
    """Find all tags containing the given text.
    Args:
        text: String to search for
        index: Index of the current sample
    Returns:
        Tuple of (image path, tag, list of all matching image-tag pairs)
    """
    global curr_tag, dir_images, filtered_df

    # Create dataframe with all images and their tags
    df = pd.DataFrame({
        'image': dir_images.image_paths,
        'tag': [get_tag(path)[0] for path in dir_images.image_paths]
    })

    # Filter to tags containing the search text
    filtered_df = df[df['tag'].str.contains(text)]

    if len(filtered_df) == 0:
        raise gr.Error(f"No samples found with \"{text}\"")

    # Ensure index is within bounds
    if index >= len(filtered_df):
        index = 0
    elif index < 0:
        index = len(filtered_df) - 1

    curr_tag = index

    return (
        filtered_df.iloc[index]['image'], 
        filtered_df.iloc[index]['tag'], 
        [(image, get_tag(image)[0]) for image in filtered_df.image]
    )


def find_next(text: str, forward: bool) -> Tuple[str, str, List[Tuple[str, str]]]:
    """Find the next/previous image-tag pair with the given text.
    Args:
        text: String to search for
        forward: True for next, False for previous
    Returns:
        Tuple of (image path, tag, list of all matching image-tag pairs)
    """
    global curr_tag, curr_keyword, filtered_df

    # If search term changed, reset index
    if curr_keyword != text:
        curr_tag = 0
        curr_keyword = text
    else:
        curr_tag = curr_tag + 1 if forward else curr_tag - 1

    return find_text_in_captions(curr_keyword, curr_tag)


def find_next_sample(text: str) -> Tuple[str, str, List[Tuple[str, str]]]:
    """Find the next image-tag pair containing the search text."""
    return find_next(text, forward=True)


def find_prev_sample(text: str) -> Tuple[str, str, List[Tuple[str, str]]]:
    """Find the previous image-tag pair containing the search text."""
    return find_next(text, forward=False)


def display_text(selected_image: gr.SelectData) -> Tuple[str, str]:
    """Return the image path and tag for the selected gallery image.
    Args:
        selected_image: Gradio SelectData event
    Returns:
        Tuple of (image path, tag)
    """
    global filtered_df, curr_tag
    curr_tag = selected_image.index
    return filtered_df.iloc[curr_tag]['image'], filtered_df.iloc[curr_tag]['tag']


def replace_text_in_caption(find: str, curr_sample_path: str, replace: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Replace text in a specific tag and save it.
    Args:
        find: Text to find
        curr_sample_path: Path to the image
        replace: Text to replace with
    Returns:
        Tuple of (updated tag, list of all matching image-tag pairs)
    """
    global filtered_df

    # Replace text in tag and save
    tag = get_tag(curr_sample_path)[0].replace(find, replace)
    save_tag(tag, get_tag_path(curr_sample_path))

    # Return updated tag and refreshed list of image-tag pairs
    return tag, [(image, get_tag(image)[0]) for image in filtered_df.image]


def replace_in_all_captions(find: str, replace: str, curr_sample_path: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Replace text in all matching tags.
    Args:
        find: Text to find
        replace: Text to replace with
        curr_sample_path: Path to current image
    Returns:
        Tuple of (current image's updated tag, list of all matching image-tag pairs)
    """
    global dir_images, filtered_df

    # Create dataframe with all images and tags
    df = pd.DataFrame({
        'image': dir_images.image_paths,
        'tag': [get_tag(path)[0] for path in dir_images.image_paths]
    })

    # Filter to tags containing the search text
    filtered_df = df[df['tag'].str.contains(find)]

    # Replace in all filtered tags and save
    for index, row in filtered_df.iterrows():
        updated_tag = row['tag'].replace(find, replace)
        save_tag(updated_tag, get_tag_path(row['image']))

    # Return current tag and refreshed list of image-tag pairs
    tag = get_tag(curr_sample_path)[0]
    return tag, [(image, get_tag(image)[0]) for image in filtered_df.image]


# ----- WebDataset Creation -----

def create_webdataset(directory_elements: List[str]) -> None:
    """Create a dataset in WebDataset format.
    Args:
        directory_elements: List of directory elements
    """
    if not directory_elements:
        raise gr.Error("Choose a directory above first.")
        
    # Create webData directory
    dir_path = os.path.join(directory_elements[0], "webData")
    os.mkdir(dir_path)
    img_paths = get_images_from_dir(directory_elements)

    gr.Info("Please wait a few moments for the dataset to be generated.")

    # Collect image-tag pairs
    combined_tag = []
    for img in img_paths:
        tag_path = get_tag_path(img)
        if os.path.exists(tag_path):
            # Copy image to webData directory
            shutil.copy(img, dir_path)
            # Get tag and add to combined data
            tag, _ = get_tag(img)
            combined_tag.append({"file_name": os.path.basename(img), "text": tag})

    # Create combined JSON file
    with open(os.path.join(dir_path, "label.json"), "w") as jsonFile:
        json.dump(combined_tag, jsonFile, indent=4)

    # Create zip archive and clean up
    shutil.make_archive(dir_path, 'zip', dir_path)
    shutil.rmtree(dir_path)

    gr.Info("Done!")


# ----- UI Helper Functions -----

def show_options(options: str) -> Dict[str, Any]:
    """Update a Gradio dropdown with a list of options.
    Args:
        options: String representation of options list
    Returns:
        Gradio update object with new choices
    """
    return gr.update(choices=ast.literal_eval(options), value=None)


def load_image_for_describe(selected_image: Optional[gr.SelectData] = None) -> Tuple[str, str, str, str]:
    """Load a selected image from the gallery into the Describe Anything image editor.
    
    Args:
        selected_image: Selected image data from the gallery
        
    Returns:
        Tuple containing (image_path, image_path, tag, tag_path)
    """
    global dir_images
    
    if dir_images is None or not dir_images.image_paths:
        raise gr.Error("No images available in the selected directory.")
    
    # If no image is selected, use the current index
    if selected_image is None:
        selected_index = dir_images.idx
    else:
        # Get the selected image path (index from the gallery)
        selected_index = selected_image.index
        
    if selected_index >= len(dir_images.image_paths):
        raise gr.Error("Invalid image selection.")
    
    selected_path = dir_images.image_paths[selected_index]
    cap_path = get_tag_path(selected_path)
    
    # Read existing tag file or create a new one
    if os.path.exists(cap_path):
        tag = read_tag(cap_path)
    else:
        open(cap_path, 'w').close()
        tag = ""
    
    # Update the current index to match the selected image
    dir_images.idx = selected_index
    
    # Return image, image_path, tag, tag_path
    return selected_path, selected_path, tag, cap_path
