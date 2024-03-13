# Text-Summarization
**Author:** [Md Rasul Islam Bapary]  
**Date:** [14.03.2024]

## Choosing a Kaggle Dataset for Large Language Model Training
In this project, I have selected the CNN-DailyMail News Text Summarization dataset from Kaggle for training or fine-tuning a Large Language Model (LLM). In this file, I will provide an overview of why I have chosen this dataset and my considerations during the selection process.
### Dataset Overview
The CNN-DailyMail dataset is one of the widely used benchmark for text summarization tasks. It consists of news articles from CNN and DailyMail along with human-generated summaries. Each article has a single highlight .The dataset is well-structured, containing a large number of articles across various topics. \
When selecting this dataset, I have looked into several factors,

    1. Data Size:
       The CNN-DailyMail dataset is sufficiently large, containing thousands of articles along with corresponding
       summaries. It contains,
       Train set : 287,113
       Validation set : 13,368
       Test set : 11,490
    2. Diversity of Content:
       This dataset is a collection of news articles covering a wide range of topics, including politics, sports, technology,
       entertainment, and more.
    3. Availability of Ground Truth Summaries:
       Each news article in the dataset comes with human-generated summaries, serving as ground truth reference points for
       evaluating the quality of the model's generated summaries.
    4. Relevance to Real-World Applications:
       News summarization is a real-world application with widespread relevance. It can be useful for tasks such as content
       recommendation systems, news aggregation platforms, and information retrieval.
    5. Potential for Creative Exploration using Prompt Engineering:
       The nature of this dataset allows for creative exploration using Generative AI also.

In conclusion, the CNN-DailyMail News Text Summarization dataset was chosen for its suitable size, relevance to real-world applications, and potential for prompt engineering experimentation. Additionally, highlighting its richness in content diversity, availability of ground truth summaries, established benchmark status, and the support it receives from the research community provides a solid foundation for training or fine-tuning Large Language Models for text summarization tasks.

## Model Selection for Text Summarization
In this section, I will discuss the selection of a relevant Kaggle model for the CNN-DailyMail News Text Summarization dataset. The chosen Kaggle model is titled "Seq2Seq Enc Dec" which can be found [here](https://www.kaggle.com/code/mohamedaref000/seq2seq-enc-dec). This model implements an Encoder-Decoder architecture with Bidirectional LSTM units, a commonly used neural network architecture for sequence-to-sequence tasks such as text summarization.

### Architecture:
**Bidirectional LSTM Encoder-Decoder:**
The use of bidirectional LSTMs allows the model to capture contextual information from both past and future tokens, enhancing its ability to understand the input text. This architecture is suitable for sequence-to-sequence tasks like text summarization. The image given below show an overview of the model architecture,
![model_architecture](https://github.com/rasul-ai/Text-Summarization/blob/4d9a0622f977d2ed82a5d0a09e51c17762fc7aaa/images/seq2seq_encoder_decoder.png)

### Relevance and Adaptation
1. Encoder-Decoder Architecture:\
The Encoder-Decoder architecture is particularly suitable for sequence-to-sequence tasks like text summarization. The encoder processes the input sequence (news article) and encodes it into a fixed-size context vector, while the decoder generates the output sequence (summary) based on this context vector. This architecture is well-aligned with the requirements of summarizing news articles into concise summaries.

2. Bidirectional LSTM Units:\
The incorporation of Bidirectional LSTM units in both the encoder and decoder enhances the model's ability to capture contextual information from both past and future tokens in the input sequence. This bidirectional context modeling can lead to more informative and coherent summaries, especially in capturing the nuanced relationships between different parts of the news articles.

3. Adaptation for CNN-DailyMail Dataset:\
To adapt the chosen model for the CNN-DailyMail dataset, several modifications and considerations can be made:
```
• Data Preprocessing:
    Preprocess the input data from the CNN-DailyMail dataset, including tokenization, padding, and truncation, to ensure
    compatibility with the model's input format.

• Model Architecture Tuning:
    Fine-tune the hyperparameters of the LSTM layers, such as the number of hidden units,layer depth, and dropout rates,
    to optimize the model's performance for summarizing news articles.

• Training Strategy:
    Employ efficient batching, distributed training, or techniques like curriculum learning to handle the diversity of
    article lengths and summary styles effectively, considering the large size of the CNN-DailyMail dataset.
```
The selected Kaggle model based on the Encoder-Decoder architecture with Bidirectional LSTM units presents a promising framework for text summarization tasks, aligning well with the requirements of the CNN-DailyMail News Text Summarization dataset. With appropriate adaptation and fine-tuning, this model holds potential for generating high-quality summaries from news articles in the dataset.

Based on the provided ROUGE scores, we can evaluate the capabilities and limitations of the chosen model as follows:

## Performance Metrics:
**ROUGE-1 Score:**
```
Recall (R): 0.104
Precision (P): 0.286
F1-Score (F): 0.146
```
**ROUGE-2 Score:**
```
Recall (R): 0.015
Precision (P): 0.036
F1-Score (F): 0.020
```
**ROUGE-L Score:**
```
Recall (R): 0.098
Precision (P): 0.271
F1-Score (F): 0.137
```
### Evaluation:
The model shows moderate performance in terms of ROUGE-1 and ROUGE-L scores, indicating its ability to generate summaries that contain overlapping n-grams with the reference summaries.\
It demonstrates relatively better precision compared to recall, suggesting that the generated summaries contain meaningful content, albeit with some brevity.

### Limitations:
The model's performance, especially in terms of ROUGE-2 score, is relatively low. This indicates a challenge in capturing longer-range dependencies and phrases that span across multiple words.\
There is a notable gap between the model's recall and precision scores, indicating room for improvement in terms of generating more comprehensive summaries without sacrificing precision.


### Potential for Further Training or Fine-Tuning:
**1. Fine-Tuning:**\
The model can be further fine-tuned by adjusting hyperparameters such as the number of LSTM units, dropout rates, and learning rates. Fine-tuning may help improve the    model's performance by optimizing its ability to capture semantic meaning and generate accurate summaries.
    
**2. Attention Mechanism:**\
Incorporating attention mechanisms within the model architecture could potentially improve its ability to focus on relevant parts of the input text when generating       summaries, leading to better performance, especially in handling longer texts.
    
**3. Transfer Learning:**\
Leveraging transfer learning techniques by initializing the model with weights pre-trained on a larger corpus or using domain-specific embeddings may help enhance
the model's generalization and performance, particularly for summarizing news articles from the CNN-DailyMail dataset.

    
In conclusion, while the chosen model demonstrates moderate performance in terms of ROUGE scores, there is still room for improvement, especially in capturing longer-range dependencies and improving recall. Further fine-tuning, architectural enhancements, and transfer learning techniques can potentially address these limitations and enhance the model's capabilities for text summarization tasks.
