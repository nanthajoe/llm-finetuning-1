# LLM Fine-Tuning for Task-Oriented Instruction Generation
This repository contains the solution for the Machine Learning Engineer take-home test. The project focuses on fine-tuning an open-source Large Language Model to specialize in generating structured, task-oriented instructions from natural language user prompts.

## 1. Choose an Open-Source LLM
For this task, the selected model is google/gemma-2b-it, a 2.5 billion parameter, instruction-tuned model from Google.

Justification:
* Performance vs. Size: Gemma 2B offers a state-of-the-art balance between high performance and manageable size. It is capable of understanding nuanced user intent while being lightweight enough to be fine-tuned efficiently on consumer-grade hardware or free cloud services like Google Colab. This makes it ideal for rapid development and cost-effective deployment.

* Instruction-Tuned Baseline: The -it (instruction-tuned) variant has already been optimized to follow commands and generate helpful, conversational responses. This provides a superior starting point, as we are not teaching the model to follow instructions from scratch but rather specializing its existing capabilities for a specific format.

* Resource Efficiency: Its smaller size directly addresses the challenge of limited computational resources. It can be loaded and fine-tuned using 4-bit quantization on a single GPU with as little as 12GB of VRAM, making the process accessible and reproducible.

* Ecosystem Support: The model is fully integrated into the Hugging Face ecosystem, with excellent support from libraries like transformers, peft, and trl, which are essential for implementing the QLoRA fine-tuning process.

## 2. Dataset Design and Preparation
*Data Format and Content*
The fine-tuning data consists of a collection of JSON objects, each representing a "prompt-completion" pair. This format cleanly separates the user's intent from the desired structured output.

Example:

{
  "prompt": "How do I update my shipping address on the app?",
  "completion": "1. Open the app and go to your Profile page.\n2. Tap on 'My Account' or 'Settings'.\n3. Select the 'Addresses' or 'Shipping Information' option.\n4. Find the address you want to update and tap 'Edit'.\n5. Make the necessary changes and tap 'Save'."
}

*Data Collection and Annotation*
A dataset of ~160 examples was created using a synthetic data generation approach. A powerful general-purpose LLM (Google's Gemini) was prompted with a list of 20 common e-commerce topics (e.g., password reset, order tracking) and instructed to generate a canonical step-by-step answer and 5-10 varied user prompts for each topic. This method allows for rapid, scalable, and high-quality data creation. The key was to generate multiple phrasings for a single intent to improve the model's real-world robustness.

*Preprocessing Steps*
1. Exploratory Data Analysis (EDA): The dataset was analyzed to understand text lengths, topic distribution, and common n-grams, ensuring data quality and balance.

2. Chat Templating: Each prompt-completion pair was formatted into a single string using the Gemma model's specific chat template (<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{completion}<end_of_turn>). This teaches the model the conversational structure of the task.

3. Data Splitting: The dataset was split into a 90% training set and a 10% validation set to monitor for overfitting during the fine-tuning process.

*Handling Potential Issues*
* Data Imbalance: By generating a similar number of examples for each of the 20 topics, the dataset was kept relatively balanced, preventing the model from becoming biased toward any single task.

* Sensitive Information: The synthetic data was generated for a fictional app ("ShopSphere"), which inherently avoids the inclusion of real user PII. For a real-world application, a sanitization step using regex or NER would be added to remove sensitive data.

* Diversity and Generalization: For each topic, multiple prompts with different phrasings, tones, and levels of detail were generated. This crucial step trains the model to generalize and recognize user intent regardless of how a question is asked.

## 3. Fine-Tuning Strategy
Approach: QLoRA (Quantized Low-Rank Adaptation)
The chosen fine-tuning approach is QLoRA, a highly efficient method from the PEFT (Parameter-Efficient Fine-Tuning) family.

Reasoning:

* Resource Efficiency: QLoRA drastically reduces computational requirements. By loading the base model in 4-bit precision and only training a small number of adapter layers, it's possible to fine-tune a multi-billion parameter model on a single GPU. This directly mitigates the challenge of limited computational resources.

* Prevents Catastrophic Forgetting: The weights of the large pre-trained model are frozen, and only the small LoRA adapters are updated. This preserves the model's vast general knowledge while skillfully adapting it to the new, specific task of generating structured instructions.

* Faster Iteration: The efficiency of QLoRA allows for much faster training cycles, enabling quicker experimentation and hyperparameter tuning.

*Key Hyperparameters*
* learning_rate (2e-4): A common and effective starting point for LoRA fine-tuning, balancing stable learning with reasonable training speed.

* r (LoRA rank, 64): Defines the size of the trainable adapter matrices. A rank of 64 provides a good trade-off between expressive power and the number of trainable parameters.

* lora_alpha (16): A scaling factor for the LoRA updates.

* num_train_epochs (1): For fine-tuning tasks with high-quality datasets, a single epoch is often sufficient to adapt the model without causing it to overfit.

* per_device_train_batch_size (4): The largest batch size that could comfortably fit within the available GPU memory.

*Mitigating Potential Challenges*
* Computational Resources: Addressed by using QLoRA and leveraging free cloud GPU services (Google Colab).

* Overfitting: Mitigated by using a validation set to monitor performance on unseen data. The Trainer was configured to evaluate every 50 steps, and the training and validation loss curves were monitored via TensorBoard to ensure both were decreasing.

* Catastrophic Forgetting: Inherently addressed by the choice of LoRA, which freezes the base model's weights.

## 4. Evaluation and Benchmarking
*Performance Metrics*
* Quantitative:

Validation Loss: The primary metric tracked during training to monitor for overfitting and ensure the model is generalizing well.

* Qualitative:

Human "Eyeball Test": The most important evaluation for this task. A human review of the generated outputs to assess correctness, clarity, formatting, and overall helpfulness.

*Benchmarking Setup*
A simple benchmark was established by comparing the outputs of two models on a set of test prompts:

1. Baseline Model: The original, non-fine-tuned google/gemma-2b-it.

2. Fine-Tuned Model: Our model after QLoRA fine-tuning.

The goal was to observe a clear improvement in the fine-tuned model's ability to consistently produce the desired numbered-list format and provide more direct, task-focused answers compared to the more conversational and sometimes verbose baseline.

Assessment Methods
* Quantitative Assessment: The final validation loss from the training run serves as the primary quantitative score.

* Qualitative Assessment: The core of the evaluation. A diverse set of 10 test prompts (including variations in phrasing and out-of-domain questions) was used to generate responses from the fine-tuned model. The outputs were manually reviewed to answer the following:

1. Correctness: Are the instructions factually correct for the given task?

2. Formatting Adherence: Does the output strictly follow the numbered-list format?

3. Clarity & Conciseness: Are the steps easy to understand and free of unnecessary conversational filler?

4. Robustness: How does the model respond to prompts it wasn't explicitly trained on or to out-of-domain questions?

The results showed a significant improvement in formatting adherence and conciseness for the fine-tuned model, confirming the success of the specialization task.