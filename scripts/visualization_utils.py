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
import numpy as np
import pandas as pd
import plotly.express as px
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from tensorflow.keras.utils import load_img, img_to_array
from keras.applications.resnet_v2 import ResNet50V2, preprocess_input


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


def create_image_tag_df(data_directory):
    """
    Creates a dataframe of all image-tag pairs in a given directory
    :param data_directory: path to the chosen directory
    :return:
    """
    df = pd.DataFrame(columns=['Image', 'tag'])
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(data_directory):
        for f in files:
            valid_types = ("png", "jpg", "jpeg")
            if f.lower().endswith(valid_types):
                image_path = os.path.join(root, f)
                cap_path = get_tag_path(image_path)
                if os.path.exists(cap_path):
                    text = read_tag(cap_path)
                    df.loc[len(df.index)] = [image_path, text]

    return df


def extract_features_img(file, model):
    """
    Extracts image features via the given model
    :param file: image file
    :param model: the model used for feature extraction
    :return: an image's representation (features)
    """
    img = load_img(file, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = model.predict(img)
    return features


def image_embeddings(df):
    """
    Feature extraction - ResNet50 V2
    :param df: dataframe of image-tag pairs
    :return: image embeddings
    """
    model = ResNet50V2()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    features = []
    for image in df.Image:
        features.append(extract_features_img(image, model))
    features = np.array(features)
    features = features.reshape(-1, 2048)

    return features


def text_embeddings(df):
    """
    Feature extraction - Sentence Transformers
    :param df: dataframe of image-tag pairs
    :return: image embeddings
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(df.tag)

    return embeddings

def pca_and_kmeans(embeddings, num_clusters):
    """
    Performs PCA on the given embeddings to reduce dimentionality, and clusters them
    :param embeddings: generated representations of image/text samples
    :param num_clusters: number of clusters
    :return: the 2D data with the cluster numbers
    """
    pca = PCA(n_components=2, random_state=24)
    pca_data = pca.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=num_clusters, n_init=5, max_iter=500, random_state=24)
    kmeans.fit(pca_data)

    return pca_data, kmeans.labels_


def create_and_plot_results_df(image_paths, cap, labels, pca_features):
    """
    Returns a Plotly plot and a dataframe with image-tag-cluster sets
    :param image_paths: list of all image paths
    :param cap: list of tags
    :param labels: list clustering labels
    :param pca_features: 2D-features for each of the samples
    :return: plot, clustered sample df
    """
    results = pd.DataFrame()
    results['Image'] = image_paths
    results['Tag'] = cap
    results['cluster'] = labels
    results['cluster'] = results['cluster'].astype(str)
    results['x'] = pca_features[:, 0]
    results['y'] = pca_features[:, 1]

    fig = px.scatter(results, x='x', y='y', color="cluster", hover_data=['Tag', 'Image'],
                     color_discrete_sequence=px.colors.qualitative.Dark24,
                     category_orders={"cluster":np.unique(labels).astype(str)})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(
        plot_bgcolor='white',
        legend_title=None
    )

    return fig, results[["Image", 'Tag', "cluster"]]
