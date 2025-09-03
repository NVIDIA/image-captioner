# Image Captioning and Visualization Tool

## Overview
This image captioning and visualization tool leverages state-of-the-art generative AI image-to-text models to automatically generate, edit, and manage captions and tags for image datasets. The tool provides a unified interface for accessing multiple vision-language models hosted by leading inference services, including [NVIDIA NIM](https://www.nvidia.com/en-us/ai/) and [Hugging Face](https://huggingface.co/), enabling users to generate high-quality image descriptions with a single click.

The tool consists of two main modules:

**Module 1: Standard Tagging** - Generate and edit image captions using various vision-language models (Kosmos-2, Llama 3.2 Vision, Gemma 3, etc.) with customizable parameters and batch processing capabilities.

**Module 2: Data Visualization & Analytics** - Comprehensive clustering and analysis tools that help users explore your image-tag datasets through interactive visualizations, keyword filtering, and word cloud generation to identify patterns and themes.

Beyond automated captioning, the tool includes powerful data management features such as bulk editing, find/replace operations, WebDataset export functionality, and interactive dataset exploration capabilities. 

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Valid API credentials for NVIDIA NIM and/or Hugging Face (depending on your preferred models)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <URL>
   cd image-captioner
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

[Module 2: Data Visualization & Analytics](#module-2-data-visualization--analytics)
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

Advanced parameters:
- max_token, default = 1024
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

## Module 2: Data Visualization & Analytics

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

 