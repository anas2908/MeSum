# Hope `The Paragraph Guy' explains the rest: Introducing MeSum, The Meme Summarizer
Code for the Paper: Hope `The Paragraph Guy' explains the rest: Introducing MeSum, The Meme Summarizer

## Abstract
Over the years, memes have evolved into multifaceted narratives on platforms like Instagram, TikTok, and Reddit, blending text, images, and audio to amplify humor and engagement. The objective of the task described in this article is to bridge the gap for individuals who may struggle to understand memes due to cultural, geographical, ancillary insights, or relevant exposure constraints, aiming to enhance meme comprehension across diverse audiences. The lack of large datasets for supervised learning and alternatives to resource-intensive vision-language models have historically hindered the development of such technology. In this work, we have made strides to overcome these challenges. We introduce "MMD," a Multimodal Meme Dataset comprising 13,494 instances, including 3,134 with audio, rendering it the largest of its kind, with 2.1 times as many samples and 9.5 times as many words in the human-annotated meme summary compared to the largest available meme captioning dataset, MemeCap. Our framework, MeSum (Meme Summariser), employs a fusion of Vision Transformer and Large Language Model technologies, providing an efficient alternative to resource-intensive Vision Language Models. Pioneering the integration of all three modalities, we attain a ROUGE-L score of 0.439, outperforming existing approaches such as zero-shot Gemini, GPT-4 Vision, LLaVA, and QwenVL, which yield scores of 0.259, 0.213, 0.177, and 0.198. We have made our codes and datasets publicly available.

## Requirements
- conda create -n CoViSum python=3.8.5
- conda activate CoViSum
- pip install -r requirement.txt


## Project Structure

This repository is organized into two main folders:

- **Baselines**: Contains the code for the QwenVL and LLaVA models.
- **Model**: Contains two key files:
  - `meme_bart_text_vid_mfccs_best`: This file is responsible for loading and preprocessing the multimodal data, including text, vision, and audio features (such as MFCCs). It prepares the inputs from different modalities and ensures they are properly formatted for model training and inference.
  - `modeling_meme_bart_anas_third`: This file defines the architecture of the model, specifically focusing on the cross-attention mechanism that allows the model to effectively fuse and attend to information from text, vision, and audio inputs. The cross-attention module plays a key role in integrating these modalities to generate accurate and coherent summaries of memes.

### Ablation Studies
Instead of creating a separate folder for ablation studies, the code is designed to allow small changes that produce different architectures. This flexible structure makes it easy to modify and experiment without duplicating files.


## Upcoming Resources

- We will soon release a video walkthrough and a text document explaining the architecture modification process in detail and code implementation.
- The model weights and the complete dataset will be made available in the same repository, accessible via an `IIT Patna` server link.

## Running Inference

To evaluate the model’s performance on your dataset:

1. Use the `meme_inference` script.
2. Load the checkpoint.
3. Provide the appropriate path to your test instance(s).

Stay tuned for updates!
