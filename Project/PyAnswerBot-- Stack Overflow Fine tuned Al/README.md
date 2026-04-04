# PyAnswerBot: Stack Overflow Fine-tuned AI

PyAnswerBot is a fine-tuned AI model trained on real Stack Overflow data to answer Python programming questions. This project demonstrates the complete machine learning pipeline from raw data collection to a working AI model using free tools like Google Colab and Hugging Face.

---

## Project Overview

PyAnswerBot takes a Python programming question as input and returns a Stack Overflow style answer. The model was fine-tuned on 4,436 high quality Python Q&A pairs filtered from the StackSample dataset available on Kaggle.

Example:

```
User: How do I reverse a list in Python?
Bot:  You can use my_list[::-1] or my_list.reverse().
      [::-1] returns a new list while .reverse() modifies the original list in place.
```

---

## Dataset

Source: StackSample 10% of Stack Overflow Q&A (Kaggle)

Files used:
- Questions.csv — Question titles and body text
- Answers.csv — Answer body text linked to questions
- Tags.csv — Tags linked to each question

Filter applied:
- Python tagged questions only
- Questions and answers with Score greater than or equal to 5
- Total training examples: 4,436 Q&A pairs

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Main programming language |
| Pandas | Data loading and processing |
| BeautifulSoup | Removing HTML tags from text |
| Hugging Face Transformers | Loading the base language model |
| Hugging Face TRL | SFTTrainer for fine-tuning |
| Google Colab T4 GPU | Free GPU for model training |
| facebook/opt-125m | Base pre-trained language model |

---

## Pipeline

```
Stack Overflow CSV Files
        |
Filter high quality Python Q&A (Score >= 5)
        |
Merge Questions with Best Answers
        |
Clean HTML tags using BeautifulSoup
        |
Convert to JSONL format
        |
Fine-tune facebook/opt-125m on T4 GPU
        |
Test and Save the model
```

---

## How to Run

Step 1: Clone the repository

```bash
git clone https://github.com/Subinthapa2092/PyAnswerBot.git
cd PyAnswerBot
```

Step 2: Install required libraries

```bash
pip install transformers datasets trl accelerate beautifulsoup4 pandas torch
```

Step 3: Download the dataset from Kaggle and place these files in the project folder:
- Questions.csv
- Answers.csv
- Tags.csv

Step 4: Open PyAnswerBot.ipynb in Google Colab with T4 GPU and run all cells.

---

## Project Structure

```
PyAnswerBot/
    PyAnswerBot.ipynb       Main Colab notebook
    training_data.jsonl     Prepared training data in JSONL format
    finetuned-model/        Saved fine-tuned model files
    README.md               Project documentation
```

---

## Training Details

| Detail | Value |
|--------|-------|
| Base Model | facebook/opt-125m |
| Training Examples | 4,436 |
| Training Epochs | 1 |
| Batch Size | 4 |
| GPU | Tesla T4 (Free on Google Colab) |
| Training Time | Approximately 25 minutes |

---

## Key Learnings

- How to clean and preprocess real world NLP data from Stack Overflow
- How to convert raw data into JSONL format for fine-tuning
- How to fine-tune a language model using Hugging Face TRL SFTTrainer
- How to use free GPU on Google Colab for model training
- How to test and save a fine-tuned language model

---

## Limitations

- The base model facebook/opt-125m is small with 125 million parameters which limits answer quality and depth
- Training was done for only 1 epoch due to resource constraints
- The model only covers Python programming questions

---

## Future Improvements

- Use a larger model like facebook/opt-350m or mistralai/Mistral-7B for better answers
- Train for more epochs to improve model performance
- Expand dataset to cover more programming languages
- Deploy as a web application using Streamlit or Gradio

---

## Author

Subin Thapa

- Website: subinthapa.com.np
- LinkedIn: linkedin.com/in/subinthapa
- GitHub: github.com/Subinthapa2092
- YouTube: Coding with Subin

---

## License

This project is open source and available under the MIT License.
