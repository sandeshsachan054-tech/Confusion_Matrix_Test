import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is not set in .env file.")

groq_client = Groq(api_key=api_key)
app = FastAPI(title="Spam Confusion Matrix Tester")

class EvaluateRequest(BaseModel):
    email_text: str
    actual_label: str

class EvaluateResponse(BaseModel):
    actual_label: str
    predicted_label: str
    outcome_code: str
    outcome_title: str
    details: str
    error_type: str

def classify_email_with_groq(text: str) -> str:
    # Bulletproof prompt and parsing
    system_prompt = (
        "You are an automated email classifier. Your task is to classify incoming text as SPAM or NOT_SPAM.\n"
        "Rules:\n"
        "- If it is spam, phishing, fake prize, gift card claim, winning notification, or urgent scam, output: SPAM\n"
        "- If it is normal, friendly, or legitimate business communication, output: NOT_SPAM\n"
        "- Reply with ONLY the single word SPAM or NOT_SPAM. No punctuation, no explanation."
    )
    
    completion = groq_client.chat.completions.create(
        model="Openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Classify this email:\n{text}"}
        ],
        temperature=0.1,
        max_tokens=50
    )
    
    raw = completion.choices[0].message.content.strip().upper()
    print(f"\n>>> [DEBUG GROQ RAW RESPONSE]: '{raw}' <<<\n")
    
    # Accurate parsing: prioritize SPAM detection unless explicitly negative
    if "NOT_SPAM" in raw or "NOT SPAM" in raw or "HAM" in raw:
        return "NOT_SPAM"
    elif "SPAM" in raw:
        return "SPAM"
    
    return "NOT_SPAM"

@app.post("/api/evaluate", response_model=EvaluateResponse)
def evaluate_matrix(payload: EvaluateRequest):
    actual = payload.actual_label.strip().upper()
    predicted = classify_email_with_groq(payload.email_text)

    if actual == "SPAM" and predicted == "SPAM":
        return EvaluateResponse(
            actual_label=actual,
            predicted_label=predicted,
            outcome_code="TP",
            outcome_title="True Positive (1)",
            details="Actually spam. Predicted spam. The filter caught the unwanted email.",
            error_type="Correct alert"
        )
    elif actual == "SPAM" and predicted == "NOT_SPAM":
        return EvaluateResponse(
            actual_label=actual,
            predicted_label=predicted,
            outcome_code="FN",
            outcome_title="False Negative (2)",
            details="Actually spam. Predicted not spam. The unwanted email reached your inbox.",
            error_type="Missed alert = Type II error"
        )
    elif actual == "NOT_SPAM" and predicted == "SPAM":
        return EvaluateResponse(
            actual_label=actual,
            predicted_label=predicted,
            outcome_code="FP",
            outcome_title="False Positive (3)",
            details="Actually not spam. Predicted spam. An important email went to the spam folder.",
            error_type="False alarm = Type I error"
        )
    else:
        return EvaluateResponse(
            actual_label=actual,
            predicted_label=predicted,
            outcome_code="TN",
            outcome_title="True Negative (4)",
            details="Actually not spam. Predicted not spam. The important email stayed in your inbox.",
            error_type="Correct pass"
        )

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Groq LLM Confusion Matrix Tester</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f4f6f9; }
            .container { max-width: 680px; margin: auto; background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            textarea { width: 100%; height: 90px; margin-top: 8px; padding: 8px; box-sizing: border-box; font-size: 14px; }
            select, button { padding: 10px; margin-top: 10px; width: 100%; font-size: 15px; }
            button { background: #007bff; color: white; border: none; cursor: pointer; border-radius: 4px; font-weight: bold; }
            .result { margin-top: 20px; padding: 15px; border-radius: 6px; display: none; }
            .TP { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
            .TN { background: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; }
            .FP { background: #fff3cd; border: 1px solid #ffeeba; color: #856404; }
            .FN { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Confusion Matrix: Groq LLM Tester</h2>
            <label><strong>Email Text:</strong></label>
            <textarea id="emailText">Congratulations! You won a $1,000 gift card. Click to claim immediately.</textarea>
            
            <label><strong>Actual Ground Truth:</strong></label>
            <select id="actualLabel">
                <option value="SPAM">Actually SPAM (+)</option>
                <option value="NOT_SPAM">Actually NOT SPAM (-)</option>
            </select>
            
            <button onclick="testEmail()">Evaluate with Groq LLM</button>
            <div id="resultBox" class="result"></div>
        </div>

        <script>
            async function testEmail() {
                const text = document.getElementById('emailText').value;
                const actual = document.getElementById('actualLabel').value;
                const box = document.getElementById('resultBox');
                box.style.display = 'block';
                box.className = 'result';
                box.innerHTML = 'Calling Groq LLM API...';

                try {
                    const response = await fetch('/api/evaluate', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ email_text: text, actual_label: actual })
                    });
                    const data = await response.json();
                    box.className = 'result ' + data.outcome_code;
                    box.innerHTML = `
                        <h3>${data.outcome_title} (${data.outcome_code})</h3>
                        <p><strong>Predicted:</strong> ${data.predicted_label} | <strong>Actual:</strong> ${data.actual_label}</p>
                        <p>${data.details}</p>
                        <p><em>Tag: ${data.error_type}</em></p>
                    `;
                } catch (err) {
                    box.innerHTML = 'Error: ' + err.message;
                }
            }
        </script>
    </body>
    </html>
    """