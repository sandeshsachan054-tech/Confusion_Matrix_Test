#  Confusion Matrix Evaluation using Groq LLM & FastAPI

An interactive full-stack AI evaluation tool designed to test, validate, and understand the 4 fundamental outcomes of a Binary Confusion Matrix (**True Positive**, **False Negative**, **False Positive**, **True Negative**) using real-time LLM inference via the Groq Cloud API.

---

##  Project Overview

In classification tasks (such as spam filtering), a model's prediction mapped against ground truth produces 4 possible scenarios:

1. **True Positive (TP):** Actually Spam → Predicted Spam *(Correct alert)*
2. **False Negative (FN):** Actually Spam → Predicted Not Spam *(Missed alert = Type II error)*
3. **False Positive (FP):** Actually Not Spam → Predicted Spam *(False alarm = Type I error)*
4. **True Negative (TN):** Actually Not Spam → Predicted Not Spam *(Correct pass)*

This application uses **FastAPI** on the backend and **Groq's Llama 3.3-70B** model to classify email text dynamically and verify how prediction errors map to standard ML performance matrices.

---

##  Tech Stack

- **Backend:** FastAPI, Uvicorn
- **LLM Engine:** Groq API (`Openai/gpt-oss-120b`)
- **Data Validation:** Pydantic
- **Frontend:** Built-in HTML5 / Vanilla JavaScript

---

##  Project Directory Structure

```text
Confusion_Matrix_Test/
├── .env                # API Keys & Secrets
├── requirements.txt    # Python dependencies
├── main.py             # FastAPI backend, LLM integration & UI
└── README.md           # Documentation
```
## Setup & Installation

1.Clone or Open the Repository:
Navigate to your project directory:
```bash
git clone https://github.com/sandeshsachan054-tech/Confusion_Matrix_Test.git
cd Confusion_Matrix_Test
```
2. Create and Activate a Virtual Environment

.Windows(CMD):
```bash
python -m venv venv
venv\Scripts\activate
```

.Windows(Powershell):
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

.Linux/macOS:
```bash
python3 -m venv venv
source env/bin/activate
```

3.Install Dependencies:
```bash
pip install -r requirements.txt
```

4. Configure Environment Variables:
​Create a file named .env in the root folder and add your Groq API key:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

## Running the Application
Start the local server using Uvicorn:
```bash
uvicorn main:app --reload --port 8000
```

​-Interactive Web UI:
```bash
 Open http://127.0.0.1:8000

​-API Swagger Documentation: 
 Open http://127.0.0.1:8000/docs
 ```

## How to Test All 4 Outcomes:

Scenario              Sample Email Input                   Ground Truth Selection             Expected Outcome

TP            "URGENT: Click here immediately to              Actually SPAM (+)               True Positive (TP)
               claim your $10,000 cash reward!"
               
FP            "Attached is the quarterly sprint               Actually NOT SPAM (-)           False Positive (FP)
               report for Review." (Triggered
               as spam)

FN            "Hey, check out this update."                   Actually SPAM (+)               False Negative (FN)
               (Phishing link missed by filter)

TN            "Doctor appointment confirmed for               Actually NOT SPAM (-)           True Negative (TN)
                this upcoming Friday."

## API Endpoint References:
Post/api/evaluate
Request Body(JSON):
```json
{
  "email_text": "You won a $1,000 gift card! Claim now.",
  "actual_label": "SPAM"
}
````

Response(JSON):
```json
{
  "actual_label": "SPAM",
  "predicted_label": "SPAM",
  "outcome_code": "TP",
  "outcome_title": "True Positive (1)",
  "details": "Actually spam. Predicted spam. The filter caught the unwanted email.",
  "error_type": "Correct alert"
}
```
