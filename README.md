News Headline Generator
A text-generation project that fine-tunes a pretrained T5-small
model to generate headlines from news articles.
Features
Fine-tunes T5-small on labelled news article/headline data
Evaluates generated headlines using ROUGE-L
Generates headlines through a simple Flask web interface
Saves the best model based on validation ROUGE-L
Tech Stack
Python, PyTorch, Hugging Face Transformers, Flask, Pandas, NumPy,
Matplotlib
Project Structure
`app.py` --- Flask app for headline generation
`headline.py` --- model training, evaluation, and test predictions
`templates/` --- HTML templates for the web interface
`data/` --- training, development, and test datasets
`Best_headline_model/` --- saved fine-tuned model and tokenizer
Run Locally
Install the required packages:
    ``` bash
    pip install flask torch transformers sentencepiece pandas numpy matplotlib rouge-score scikit-learn tqdm
    ```
Run the training script to create `Best_headline_model/`:
    ``` bash
    python headline.py
    ```
Start the web app:
    ``` bash
    python app.py
    ```
Open the local URL shown in the terminal.
> The `Best_headline_model/` folder must exist before starting `app.py`.
