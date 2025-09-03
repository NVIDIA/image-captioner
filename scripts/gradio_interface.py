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

# Standard library imports
import os

# Third-party imports
import gradio as gr

# Local imports
from tagging_utils import *

# NVIDIA branding theme for the interface
NVIDIA_GREEN = '#76b900'
theme = gr.themes.Default(
    primary_hue="lime",
    neutral_hue="neutral",
).set(
    button_secondary_background_fill_hover_dark=NVIDIA_GREEN,
    slider_color_dark=NVIDIA_GREEN,
    accordion_text_color_dark=NVIDIA_GREEN,
    checkbox_background_color_selected_dark=NVIDIA_GREEN,
    border_color_accent_dark=NVIDIA_GREEN,
    button_secondary_background_fill_hover=NVIDIA_GREEN,
    slider_color=NVIDIA_GREEN,
    accordion_text_color=NVIDIA_GREEN,
    checkbox_background_color_selected=NVIDIA_GREEN,
    border_color_accent=NVIDIA_GREEN
)


def create_tagging_tab():
    """Create the 'Tag Dataset' tab with all components and event handlers."""
    with gr.Tab("Tag Dataset"):
        # Directory selection components
        with gr.Accordion("Select data directory"):
            chosen_dir = gr.FileExplorer(
                interactive=True,
                root_dir=os.path.expanduser('~'),
                label="Supported image formats: png, jpg, jpeg",
                show_label=True,
                ignore_glob='*/.*'
            )

        with gr.Row():
            select_button = gr.Button("Select", scale=1)
            path_box = gr.Textbox(label="Path", interactive=False, show_label=False, scale=4)

        directory_gallery = gr.Gallery(label="Gallery View", rows=1, columns=10, height="4cm", show_label=False)

        with gr.Row():
            prev_button = gr.Button("Previous")
            next_button = gr.Button("Next")

        # Main tagging controls area
        with gr.Row():
            # Left column - controls
            with gr.Column(scale=1):
                with gr.Row():
                    model_choice = gr.Dropdown(
                        choices=list(model_host_mapping.keys()),
                        label="Choose a model",
                        interactive=True
                    )
                    host_service = gr.Textbox(label="Hosting Service", interactive=False)

                api_key = gr.Text(
                    label="Enter your API key",
                    interactive=True,
                    type='password',
                    info="Can be generated from the corresponding platform."
                )
                advanced_params = gr.Text(
                    label="Advanced parameters",
                    interactive=True,
                    info="Should be comma separated. Refer to docs for parameters."
                )

                long_or_short_tag = gr.Radio(
                    choices=["Long", "Short"],
                    label="Tag length:",
                    value="Short",
                    info="long: 100 tokens, short: 25 tokens (supported for models via NVIDIA NIM)"
                )

                with gr.Row():
                    gen_all_button = gr.Button("Generate for all images")
                    gen_button = gr.Button("Generate")

                gr.Markdown(
                    "If interested in adding a prefix/suffix only to the current tag, please do so in the Tag box."
                )

                with gr.Row():
                    prefix = gr.Textbox(label="Prefix (optional)")
                    suffix = gr.Textbox(label="Suffix (optional)")

                prefix_suffix_button = gr.Button("Add to all tags")

            # Right column - image and tag display
            with gr.Column(scale=1):
                image = gr.Image(label="Output", show_label=False, show_download_button=False, scale=2, height="2vh")
                image_path = gr.Text(label="Image Path", visible=False)
                tag = gr.Textbox(label="Tag", interactive=True)
                tag_path = gr.Text(label="tag Path", visible=False)
                gen_tag = gr.Textbox(label="Generated tag", interactive=False, show_copy_button=True)
                save_button = gr.Button("Save")

        web_data_button = gr.Button("Export data in WebDataset format")

        output = [image, image_path, tag, tag_path, gen_tag]

        # Connect event handlers
        select_button.click(
            fn=select_directory,
            inputs=chosen_dir,
            outputs=output + [path_box, directory_gallery],
            api_name="select"
        )

        next_button.click(
            fn=show_next_image,
            inputs=None,
            outputs=output,
            api_name="next"
        )

        prev_button.click(
            fn=show_prev_image,
            inputs=None,
            outputs=output,
            api_name="prev"
        )

        gen_button.click(
            fn=gen_tag_from_model,
            inputs=[model_choice, host_service, image_path, api_key, advanced_params, long_or_short_tag],
            outputs=gen_tag
        )

        gen_all_button.click(
            fn=gen_tag_all,
            inputs=[model_choice, host_service, api_key, image_path, advanced_params, long_or_short_tag],
            outputs=tag
        )

        save_button.click(
            fn=save_tag,
            inputs=[tag, tag_path],
            outputs=None
        )

        prefix_suffix_button.click(
            fn=add_pre_and_suffix,
            inputs=[prefix, suffix, image_path],
            outputs=tag
        )

        web_data_button.click(
            fn=create_webdataset,
            inputs=[chosen_dir],
            outputs=None
        )

        model_choice.change(
            fn=update_host,
            inputs=[model_choice],
            outputs=[host_service]
        )

        with gr.Accordion("Find and Replace", open=False):
            gr.Markdown("Case-sensitive")
            
            with gr.Column():
                with gr.Row():
                    find_text = gr.Textbox(placeholder="Find", show_label=False)
                    replace_text = gr.Textbox(placeholder="Replace with", show_label=False)
                with gr.Row():
                    find_button = gr.Button("Find")
                    replace_button = gr.Button("Replace")
                    replace_all_button = gr.Button("Replace All")
                with gr.Column():
                    find_and_replace_gallery = gr.Gallery(label="Gallery", show_label=False, rows=1, columns=10)
                    sample_path = gr.Textbox(show_copy_button=True, label="path")
                    sample_tag = gr.Textbox(show_label=False)
                    with gr.Row():
                        prev_find_button = gr.Button("Previous")
                        next_find_button = gr.Button("Next")

        # Connect find & replace event handlers
        find_button.click(
            fn=find_next_sample,
            inputs=[find_text],
            outputs=[sample_path, sample_tag, find_and_replace_gallery]
        )

        replace_button.click(
            fn=replace_text_in_caption,
            inputs=[find_text, sample_path, replace_text],
            outputs=[sample_tag, find_and_replace_gallery]
        )

        replace_all_button.click(
            fn=replace_in_all_captions,
            inputs=[find_text, replace_text, sample_path],
            outputs=[sample_tag, find_and_replace_gallery]
        )

        next_find_button.click(
            fn=find_next_sample,
            inputs=[find_text],
            outputs=[sample_path, sample_tag, find_and_replace_gallery]
        )

        prev_find_button.click(
            fn=find_prev_sample,
            inputs=[find_text],
            outputs=[sample_path, sample_tag, find_and_replace_gallery]
        )

        find_and_replace_gallery.select(
            display_text, 
            inputs=None, 
            outputs=[sample_path, sample_tag]
        )
        
        return chosen_dir


def create_visualization_tab(chosen_dir_tagging):
    """Create the 'Visualize Data' tab with all components and event handlers."""
    with gr.Tab("Visualize Data"):
        gr.Markdown("""
                    # Clustering
                    Cluster the data from the Tag Dataset directory.
                    """)
        
        # Clustering controls
        image_or_text = gr.Radio(
            choices=["Images", "Tags"], 
            label="Cluster based on:", 
            value="Tags",
            info="Image based clustering might take a little longer. It will only include tagged images."
        )
        num = gr.Slider(label="Number of Clusters", minimum=1, step=1)
        plot = gr.Plot(label="Clusters", show_label=False)

        load_button = gr.Button("Load")
        
        load_button.click(
            fn=cluster_and_plot,
            inputs=[chosen_dir_tagging, num, image_or_text],
            outputs=plot
        )

        # Filtering section
        with gr.Column():
            gr.Markdown("""
                       ## Filter images
                       Enter a cluster number to view all image-tag pairs in that cluster, 
                       OR filter based on comma-separated keywords (no spaces).
                       
                       Make sure the number box is empty before filtering based on keywords.
                       
                       Once you tweak a caption, load the filtered samples again to see the updated text.
                       """)

            with gr.Row():
                cluster_number = gr.Number(label="Cluster number", interactive=True, value=None, minimum=0)
                keywords = gr.Textbox(label="Keywords")
                
            load_filtered_button = gr.Button("Load filtered samples")
            filtered_gallery = gr.Gallery(label="Cluster", columns=5, show_label=False)

            with gr.Row():
                cluster_image_paths_dropdown_options = gr.Textbox(label="Image paths", visible=False)
                cluster_image_paths_dropdown = gr.Dropdown(label="Image paths dropdown", interactive=True)
                load_tag_button = gr.Button("Load tag")

            with gr.Row():
                chosen_image_tag = gr.Text(label="Tag", interactive=True)
                chosen_tag_path = gr.Text(visible=False)
                save_tag_changes_button = gr.Button("Save changes")

        load_filtered_button.click(
            fn=load_filtered_grid,
            inputs=[cluster_number, keywords],
            outputs=[filtered_gallery, cluster_image_paths_dropdown_options]
        )

        load_tag_button.click(
            fn=get_tag,
            inputs=[cluster_image_paths_dropdown],
            outputs=[chosen_image_tag, chosen_tag_path]
        )

        save_tag_changes_button.click(
            fn=save_tag,
            inputs=[chosen_image_tag, chosen_tag_path],
            outputs=None
        )

        cluster_image_paths_dropdown_options.change(
            fn=show_options,
            inputs=[cluster_image_paths_dropdown_options],
            outputs=[cluster_image_paths_dropdown]
        )

        filtered_gallery.select(
            display_filter_sample, 
            inputs=None, 
            outputs=[cluster_image_paths_dropdown]
        )

        gr.Markdown("""
                    ---         
                    ## Generate Word Clouds for the clusters to understand them better.
                    """)
                    
        with gr.Row():
            with gr.Column():
                cluster_number_wc = gr.Number(label="Cluster number", interactive=True)
                gen_wordcloud_button = gr.Button("Load Word Cloud")
                dataframe = gr.DataFrame(
                    col_count=2, 
                    label="Tag frequencies", 
                    headers=["Word", "Frequency"]
                )
            word_cloud = gr.Plot(label="Word Cloud", show_label=False)

        gen_wordcloud_button.click(
            fn=gen_wordcloud,
            inputs=[cluster_number_wc],
            outputs=[word_cloud, dataframe]
        )




def page():
    """
    Defines the layout of the interface with different tabs and components.
    :return: the configured Gradio interface
    """
    with gr.Blocks(theme=theme) as demo:
        # Create the two main tabs
        chosen_dir_tagging = create_tagging_tab()
        
        # Pass the tagging directory to the visualization tab
        create_visualization_tab(chosen_dir_tagging)

    return demo


if __name__ == "__main__":
    interface = page()
    interface.launch(show_error=True)
