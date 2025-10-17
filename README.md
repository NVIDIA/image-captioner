# Image Captioning and Visualization Tool

## Overview
This image captioning and visualization tool leverages state-of-the-art generative AI image-to-text models to automatically generate, edit, and manage captions and tags for image datasets. The tool provides a unified interface for accessing multiple vision-language models hosted by leading inference services, including [NVIDIA NIM](https://www.nvidia.com/en-us/ai/) and [Hugging Face](https://huggingface.co/), enabling users to generate high-quality image descriptions with a single click.

The tool consists of three main modules:

**Module 1: Standard Captioning** - Generate and edit image captions using various vision-language models (Kosmos-2, Llama 3.2 Vision, Gemma 3, etc.) with customizable parameters and batch processing capabilities.

**Module 2: Describe Anything** - Advanced localized captioning that allows users to select specific regions within images and generate detailed descriptions for those areas, perfect for datasets requiring precise object or scene descriptions for downstream applications.

**Module 2: Data Visualization & Analytics** - Comprehensive clustering and analysis tools that help users explore your image-caption datasets through interactive visualizations, keyword filtering, and word cloud generation to identify patterns and themes.

Beyond automated captioning, the tool includes powerful data management features such as bulk editing, find/replace operations, WebDataset export functionality, and interactive dataset exploration capabilities. 

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Valid API credentials for NVIDIA NIM and/or Hugging Face (depending on your preferred models)
- NVIDIA GPU required for Module 2

### Installation Steps

1. **Clone the repository with submodules**
   
   **Option A: Clone with submodules flag**
   ```bash
   git clone --recurse-submodules <URL>
   cd image-captioner
   ```
   **If the dam folder appears empty after cloning:**
   ```bash
   git submodule update --init --recursive
   ```

2. **Set up a virtual environment**

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application**
   ```bash
   cd scripts
   python gradio_interface.py
   ```

5. **Access the web interface**
   - Open your web browser and navigate to the local URL displayed in the terminal

### Sample Dataset
For experimentation, a sample image dataset can be downloaded from [here](https://www.kaggle.com/datasets/adityajn105/flickr8k/data?select=Images). Alternatively, any image or image-text dataset can be used.

# Detailed User Guide
## Table Of Contents
[Module 1: Tagging](#module-1-tagging)
* [Pipeline](#pipeline)
* [Supported Models and Advanced Parameters](#supported-models)
* [Adding a new model](#adding-support-for-a-new-model)

[Module 2: Tagging via Describe Anything](#module-2-tagging-via-describe-anything)
* [Pipeline](#pipeline-1)
* [API Requirements](#api-requirements)

[Module 3: Data Visualization & Analytics](#module-3-data-visualization--analytics)
* [Overview](#overview)
* [Visualizing your dataset](#steps)
## Module 1: Tagging
### Pipeline
1. Choose a directory

Choose a directory with image samples, and click on the 'Select' button.
The selected directory will be visible in the adjacent text box.

2. Browse through images

Browse through the images using the next and previous buttons.

3. Editing/writing tags

For each image, the corresponding tag file's contents are read and rendered in the tag text box.
If there is no tag file, a new blank text file is created.
Users can edit these tags and save them using the button. Alternatively, a new tag can be generated, as described in the next section.

4.  Generating new tags

For each image, a new tag can be generated using one of the supported models. Make sure to use a valid API key. Optionally, advanced parameters can be passed to the model as well, if supported. 
These parameters should be comma separated, and use "=" to assign values.

**Sample argument:** _max_tokens=512, temperature=0.30_

Before generating tags, users can choose between the two specified lengths, to prompt the model accordingly.

For a specific type of output (like comma separated tags instead of sentences), 
the prompt can be tweaked in ```model_call.py```. This feature is not supported for Hugging Face.

5.  Adding prefixes/suffixes

Optionally, strings can be added before/after each tag's text. 
Please note that these changes are applied to all tags. If a prefix and/or suffix needs to be added
to a particular sample only, it can be done by editing it directly.

6.  Exporting the dataset in WebDataset format.

The loaded dataset can be exported in the WebDataset format. This creates a webdata.zip file which contains all the images along with a json file listing out the image-tag pairs.

7.  Using the find/replace feature

Users can also search for specific words/phrases and replace them with the desired keywords. 

### Supported models
#### NVIDIA NIM

1. [Kosmos-2](https://build.nvidia.com/microsoft/microsoft-kosmos-2)
2. [Llama 3.2 Vision 90B](https://build.nvidia.com/meta/llama-3.2-90b-vision-instruct)
3. [NVIDIA Vila](https://build.nvidia.com/nvidia/vila)
4. [Llama-3.1 Nemotron Nano VL 8B](https://build.nvidia.com/nvidia/llama-3.1-nemotron-nano-vl-8b-v1)

Advanced parameters:
- max_tokens, default = 1024
- temperature, default = 0.20
- top_p, default = 0.20

#### Hugging Face
1. [Gemma 3](https://huggingface.co/google/gemma-3-12b-it)

Advanced parameters: None

### Adding support for a new model

To add an **image-to-text** model available on the currently supported inference services:
1. Add a model to host mapping in ```model_host_mapping``` in ```tagging_utils.py```. 
2. In the ```gen_tag_from_model``` method, add an ```elif``` condition under the corresponding host, along with the model's URL and other required parameters.  
3. Re-run the gradio_interface.py file to make calls to this newly supported model via the user interface.

## Module 2: MOSAIC - Tagging via Describe Anything

The Describe Anything tab provides advanced localized captioning capabilities for generating detailed descriptions of specific regions within images. This module extends the core Describe Anything pipeline to support multi-region descriptions and comprehensive caption generation.

**This module replicates the core functionality of Module 1 with a new caption generation approach.**

### Pipeline

1. Choosing a directory
Follow the same steps as in the tagging tab.

2. Selecting regions for description

Click and drag on the image to select regions of interest. Multiple regions can be selected for comprehensive description generation.

3. Generating localized descriptions

For each selected region, detailed descriptions can be generated using the Describe Anything model. The model provides context-aware captions that describe the specific content within the selected areas.

4. Editing and saving descriptions

Generated descriptions can be edited and saved. The system maintains the relationship between image regions and their corresponding descriptions.

### API Requirements
- An API token is required from either [OpenAI](https://platform.openai.com/) or [build.nvidia.com](https://build.nvidia.com/)

## Module 3: Data Visualization & Analytics

After finalizing image-tag pairs, this module can be accessed by clicking on **Visualize Data**. It provides comprehensive clustering and analysis tools to explore your image-tag datasets through interactive visualizations, keyword filtering, and word cloud generation.

### Technical Overview

The clustering implementation uses the following components:
- **Image Features**: ResNet50 V2 for extracting visual representations
- **Text Features**: Sentence BERT for semantic text embeddings
- **Dimensionality Reduction**: PCA for efficient processing
- **Clustering Algorithm**: K-Means for grouping similar samples

### Workflow

#### 1. Clustering Analysis
- **Choose clustering target**: Select whether to cluster images or tags (default: tags)
- **Set cluster count**: Use the slider or text input to specify the number of clusters
- **Generate clusters**: Click **Load** to create an interactive cluster plot
- **Explore results**: Hover over data points to view individual sample tags

#### 2. Cluster Exploration
- **Select cluster**: Enter a cluster number and click **Load Cluster Contents**
- **Browse samples**: View all images in the selected cluster through an interactive gallery
- **Review tags**: Each image's associated tag is displayed in a side panel
- **Edit tags**: Use the dropdown menu to select and edit specific image tags

#### 3. Keyword Filtering
- **Enter keywords**: Provide comma-separated keywords in the search box
- **Filter results**: The gallery updates to show only samples matching your keywords
- **Note**: Ensure no cluster number is specified when using keyword filtering

#### 4. Word Cloud Generation
- **Select cluster**: Enter a cluster number for analysis
- **Generate visualization**: Click **Load Word Cloud** to create a word cloud
- **Analyze themes**: View the most prominent terms and their frequencies in the cluster
- **Review data**: Access a detailed list of tags with their occurrence counts 

## Acknowledgements

This project's **MOSAIC - Describe Anything** module builds upon the core Describe Anything pipeline from the [Describe Anything repository](https://github.com/NVlabs/describe-anything) by **NVIDIA**, **UC Berkeley**, and **UCSF**. The original pipeline provides the foundation for detailed localized captioning, which we have extended to support multi-region descriptions and comprehensive caption generation for our image tagging tool. 