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
import torch

# Local imports
from tagging_utils import *
from DAM_utils import *

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
                    info="long: 150 tokens, short: 75 tokens (supported for models via NVIDIA NIM)"
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


def create_visualization_tab(chosen_dir_tagging, chosen_dir_describe):
    """Create the 'Visualize Data' tab with all components and event handlers."""
    with gr.Tab("Visualize Data"):
        gr.Markdown("""
                    # Clustering
                    Select which directory to visualize, then cluster the data.
                    """)
        
        # Directory source selection
        dir_source = gr.Radio(
            choices=["Tag Dataset directory", "Describe Anything directory"],
            label="Choose directory source:",
            value="Tag Dataset directory"
        )
        
        # Display current directory path
        current_dir_path = gr.Textbox(label="Current directory path", interactive=False)
        
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
        
        # Update directory source handler
        def update_dir_source(source):
            # This will be handled in the cluster_and_plot function
            return f"Using directory from: {source}"
            
        dir_source.change(
            fn=update_dir_source,
            inputs=[dir_source],
            outputs=[current_dir_path]
        )
        
        # Modified cluster_and_plot function call
        def cluster_with_dir_source(dir_source, chosen_dir_tagging, chosen_dir_describe, num, image_or_text):
            chosen_dir = chosen_dir_tagging if dir_source == "Tag Dataset directory" else chosen_dir_describe
            return cluster_and_plot(chosen_dir, num, image_or_text)
            
        load_button.click(
            fn=cluster_with_dir_source,
            inputs=[dir_source, chosen_dir_tagging, chosen_dir_describe, num, image_or_text],
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

        # Modified load_filtered_grid function call
        def load_filtered_with_dir_source(dir_source, chosen_dir_tagging, chosen_dir_describe, cluster_number, keywords):
            chosen_dir = chosen_dir_tagging if dir_source == "Tag Dataset directory" else chosen_dir_describe
            return load_filtered_grid(cluster_number, keywords)
            
        load_filtered_button.click(
            fn=load_filtered_with_dir_source,
            inputs=[dir_source, chosen_dir_tagging, chosen_dir_describe, cluster_number, keywords],
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

        # Modified gen_wordcloud function call
        def gen_wordcloud_with_dir_source(dir_source, chosen_dir_tagging, chosen_dir_describe, cluster_number_wc):
            chosen_dir = chosen_dir_tagging if dir_source == "Tag Dataset directory" else chosen_dir_describe
            return gen_wordcloud(cluster_number_wc)
            
        gen_wordcloud_button.click(
            fn=gen_wordcloud_with_dir_source,
            inputs=[dir_source, chosen_dir_tagging, chosen_dir_describe, cluster_number_wc],
            outputs=[word_cloud, dataframe]
        )


def create_describe_anything_tab():
    """Create the 'Describe Anything' tab with all components and event handlers."""
    with gr.Tab("Describe Anything"):
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
            
        with gr.Row():
            # Left column - image input and controls
            with gr.Column():
                image_input = gr.ImageEditor(
                    type="pil", 
                    sources=[], 
                    brush=gr.Brush(colors=["#000000"], color_mode="fixed", default_size=20),
                    eraser=False,
                    layers=False,
                    transforms=[]
                )
                query = gr.Textbox(
                    label="Prompt", 
                    value="<image>\nDescribe the masked region in detail.", 
                    visible=False
                )
                submit_btn = gr.Button("Describe", variant="primary")
                
                # Original local caption
                original_caption = gr.Textbox(
                    label="Existing caption",
                    visible=True,
                    interactive=False,
                    show_copy_button=True
                )

                # Prefix/Suffix
                with gr.Row():
                    prefix = gr.Textbox(label="Prefix (optional)")
                    suffix = gr.Textbox(label="Suffix (optional)")
                
                prefix_suffix_button = gr.Button("Add to all captions")

            # Right column - output
            with gr.Column():
                output_image = gr.Image(label="Image with Region", visible=True)
                description = gr.Textbox(label="Region Descriptions", visible=True, lines=7)
                
                # Caption generation section
                with gr.Group():
                    with gr.Row():
                        api_key_input = gr.Textbox(
                            label="API Key (Required for caption generation)", 
                            interactive=True, 
                            visible=True,
                            type='password'
                        )
                        model_choice_caption = gr.Dropdown(
                            choices=["GPT-4o (OpenAI)", "VILA (NVIDIA)", "Gemma 3 (Hugging Face)"],
                            label="Choose Model",
                            value="VILA (NVIDIA)",
                            interactive=True
                        )
                    generate_caption_btn = gr.Button("Generate Caption", variant="primary")
                    combined_caption = gr.Textbox(
                        label="Generated Caption", 
                        visible=True, 
                        lines=5, 
                        interactive=True,
                        show_copy_button=True
                    )
                    save_button = gr.Button("Save")
                
                # State variables for storing processed data
                region_descriptions_state = gr.State([])
                processed_image_state = gr.State(None)
                image_path = gr.Text(label="Image Path", visible=False)
                tag_path = gr.Text(label="Tag Path", visible=False)

        # WebDataset export
        web_data_button = gr.Button("Export data in WebDataset format")

        # Find and Replace section
        with gr.Accordion("Find and Replace", open=False):
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

        # Connect event handlers
        select_button.click(
            fn=select_directory,
            inputs=chosen_dir,
            outputs=[image_input, image_path, original_caption, tag_path, description, path_box, directory_gallery],
            api_name="select_describe"
        )

        next_button.click(
            fn=show_next_image,
            inputs=None,
            outputs=[image_input, image_path, original_caption, tag_path, description],
            api_name="next_describe"
        )

        prev_button.click(
            fn=show_prev_image,
            inputs=None,
            outputs=[image_input, image_path, original_caption, tag_path, description],
            api_name="prev_describe"
        )

        submit_btn.click(
            fn=describe,
            inputs=[image_input, query, image_path],
            outputs=[output_image, description, region_descriptions_state, processed_image_state]
        )
        
        generate_caption_btn.click(
            fn=generate_caption,
            inputs=[api_key_input, model_choice_caption, region_descriptions_state, processed_image_state],
            outputs=combined_caption
        )
        
        save_button.click(
            fn=save_tag,
            inputs=[combined_caption, tag_path],
            outputs=[original_caption]
        )
        
        prefix_suffix_button.click(
            fn=add_pre_and_suffix,
            inputs=[prefix, suffix, image_path],
            outputs=original_caption
        )
        
        web_data_button.click(
            fn=create_webdataset,
            inputs=[chosen_dir],
            outputs=None
        )
        
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
        
        # Also connect directory_gallery to load image to image_input
        directory_gallery.select(
            fn=lambda index=None: load_image_for_describe(index),
            outputs=[image_input, image_path, original_caption, tag_path]
        )
        
        return chosen_dir


def initialize_models():
    """Initialize and set up the models required for the Describe Anything tab."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load SAM model and processor
    sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    # Disable torch initialization to avoid reinitialization
    disable_torch_init()

    # Initialize DAM model
    dam = DescribeAnythingModel(
        model_path="nvidia/DAM-3B", 
        conv_mode="v1", 
        prompt_mode="full+focal_crop", 
    )

    # Make models accessible globally via the describe_anything module
    import DAM_utils as DAM_utils
    DAM_utils.sam_model = sam_model
    DAM_utils.sam_processor = sam_processor
    DAM_utils.dam = dam
    DAM_utils.device = device


def page():
    """
    Defines the layout of the interface with different tabs and components.
    :return: the configured Gradio interface
    """
    with gr.Blocks(theme=theme) as demo:
        # Create the three main tabs
        chosen_dir_tagging = create_tagging_tab()
        chosen_dir_describe = create_describe_anything_tab()
        
        # Pass both directories to the visualization tab
        create_visualization_tab(chosen_dir_tagging, chosen_dir_describe)

    # Initialize the models for the Describe Anything tab
    initialize_models()

    return demo


if __name__ == "__main__":
    interface = page()
    interface.launch(show_error=True)
