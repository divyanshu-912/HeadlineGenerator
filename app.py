from pathlib import Path

from flask import Flask, render_template, request
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

app = Flask(__name__)

# Always resolve the model path relative to this app.py file.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Best_headline_model"

if not MODEL_PATH.is_dir():
    raise FileNotFoundError(
        f"Trained model folder not found: {MODEL_PATH}\n"
        "Run headline.py successfully first so Best_headline_model is created."
    )

print(f"Loading trained model from: {MODEL_PATH}")

# local_files_only=True prevents Transformers from accidentally
# treating the local folder name as a Hugging Face repository.
tokenizer = T5Tokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
)
model = T5ForConditionalGeneration.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)
model.eval()


def generate_headline(article):
    input_text = "summarize: " + article

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=50,
            num_beams=5,
            early_stopping=True
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


@app.route("/", methods=["GET", "POST"])
def home():
    headline = None

    if request.method == "POST":
        article = request.form.get("article", "")

        if article.strip():
            headline = generate_headline(article)

    return render_template("index.html", headline=headline)


if __name__ == "__main__":
    app.run(debug=True)
