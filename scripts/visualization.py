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
from visualization_utils import *


def create_clusters_and_plot(data_directory, num_clusters, tag_based):
    """
    Clusters the images/tags in the directory into the specified # of clusters.
    Dimensionality reduction: PCA
    Clustering: K-means
    :param data_directory: chosen directory
    :param num_clusters: number of clusters
    :return: final plot and df of image-tag-cluster sets
    """

    if not data_directory:
        raise gr.Error("Choose a directory in the chosen tab")

    num_clusters = int(num_clusters)
    data_directory = data_directory[0]
    df = create_image_tag_df(data_directory)

    if len(df['Image']) < num_clusters:
        raise gr.Error("The number of clusters is greater than the number of tagged samples.")

    # tag-based clustering
    if tag_based:
        embeddings = text_embeddings(df)

    # image-based clustering
    else:
        embeddings  = image_embeddings(df)


    dim_reduced_data, labels = pca_and_kmeans(embeddings, num_clusters)
    plot, image_tag_clusters = create_and_plot_results_df(df.Image, df.tag, labels, dim_reduced_data)

    return plot, image_tag_clusters 
