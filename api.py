import pickle
import numpy as np
from collections import defaultdict
from textblob import TextBlob
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load your Q-table
with open("q_table3.pkl", "rb") as f:
    q_table_data = pickle.load(f)

class QTable(defaultdict):
    def __init__(self, n_actions):
        super().__init__(lambda: np.zeros(n_actions))

# Rebuild QTable
n_actions = 5  # number of strategies
q_table = QTable(n_actions)
q_table.update(q_table_data)

# Your strategies
STRATEGIES = {
    0: "Friendly Reminder",
    1: "Firm Reminder",
    2: "Assign to Telecaller",
    3: "Assign to Agent",
    4: "Legal Notification"
}

# Customer data
customer_data = [
    {"phoneNumber": "917891425360", "overdue_days": 22, "missed_payments": 3, "risk_category": 2, "demo": 1, "income": 20000},
    {"phoneNumber": "917995827651", "overdue_days": 5, "missed_payments": 0, "risk_category": 0, "demo": 0, "income": 50000},
    {"phoneNumber": "919182574071", "overdue_days": 17, "missed_payments": 2, "risk_category": 1, "demo": 2, "income": 30000},
    {"phoneNumber": "2223334444", "overdue_days": 10, "missed_payments": 1, "risk_category": 0, "demo": 1, "income": 45000},
    {"phoneNumber": "3334445555", "overdue_days": 20, "missed_payments": 1, "risk_category": 2, "demo": 2, "income": 35000},
    {"phoneNumber": "4445556666", "overdue_days": 12, "missed_payments": 0, "risk_category": 1, "demo": 0, "income": 60000},
    {"phoneNumber": "5556667777", "overdue_days": 8, "missed_payments": 0, "risk_category": 0, "demo": 2, "income": 75000},
    {"phoneNumber": "6667778888", "overdue_days": 5, "missed_payments": 0, "risk_category": 0, "demo": 1, "income": 50000},
    {"phoneNumber": "7778889999", "overdue_days": 25, "missed_payments": 3, "risk_category": 2, "demo": 0, "income": 20000},
    {"phoneNumber": "8889990000", "overdue_days": 7, "missed_payments": 1, "risk_category": 1, "demo": 1, "income": 48000},
    {"phoneNumber": "9990001111", "overdue_days": 15, "missed_payments": 1, "risk_category": 1, "demo": 2, "income": 55000},
    {"phoneNumber": "0001112222", "overdue_days": 2, "missed_payments": 0, "risk_category": 0, "demo": 0, "income": 32000},
    {"phoneNumber": "1112223333", "overdue_days": 28, "missed_payments": 4, "risk_category": 2, "demo": 1, "income": 18000},
    {"phoneNumber": "2223334444", "overdue_days": 14, "missed_payments": 0, "risk_category": 1, "demo": 2, "income": 46000}
]

negative_keywords = ['harass', 'stop', 'sue', 'legal', 'complain', 'angry', 'annoy', 'report']

def analyze_customer_message(message):
    sentiment = TextBlob(message).sentiment.polarity
    has_neg = any(kw in message.lower() for kw in negative_keywords)
    if has_neg:
        sentiment -= 0.3
    return round(max(-1.0, min(sentiment, 1.0)), 2), has_neg

def discretize_state(state):
    overdue = int(state[0] > 15)
    missed = int(state[1])
    risk = int(state[2])
    sentiment = round(state[3] * 2) / 2
    demo = int(state[4])
    income = int(round(state[5] / 10000) * 10000)
    distress = round(state[6] * 2) / 2
    empathy = round(state[7] * 2) / 2
    return (overdue, missed, risk, sentiment, demo, income, distress, empathy)

def build_state_from_message(message, overdue_days, missed_payments, risk_category, demo, income, distress, empathy):
    sentiment, _ = analyze_customer_message(message)
    return np.array([overdue_days, missed_payments, risk_category, sentiment, demo, income, distress, empathy])

def predict_strategy(state_vector, qtable, default="Friendly Reminder"):
    state_key = discretize_state(state_vector)
    if state_key in qtable:
        q_values = qtable[state_key]
        ranked = np.argsort(q_values)[::-1]
        best_action = int(ranked[0])
        return STRATEGIES[best_action], [STRATEGIES[i] for i in ranked]
    else:
        return default, list(STRATEGIES.values())

# --- FastAPI setup ---
app = FastAPI()

class WebhookRequest(BaseModel):
    messageType: str
    message: str
    phoneNumber: str
    summary_text: str | None = None
    distress: float
    empathy: float

@app.get("/")
def root():
    return {"message": "Hello from Colab + FastAPI + ngrok!"}

@app.post("/rlmessage")
async def webhook(req: WebhookRequest):
    # Lookup customer data
    customer = next((c for c in customer_data if c["phoneNumber"] == req.phoneNumber), None)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    # Build state
    state_vec = build_state_from_message(
        message=req.message,
        overdue_days=customer["overdue_days"],
        missed_payments=customer["missed_payments"],
        risk_category=customer["risk_category"],
        demo=customer["demo"],
        income=customer["income"],
        distress=req.distress,
        empathy=req.empathy
    )

    templates = {
        "Friendly Reminder": (
            "You are a customer support assistant for a financial services provider.\n"
            "Write a polite and respectful message to a known customer reminding them of their overdue payment.\n"
            "- Keep the tone warm but professional, not overly casual.\n"
            "- Mention that payment is overdue and offer help if needed.\n"
            "- Limit to 2–3 sentences and use newlines for readability."
        ),
        "Firm Reminder": (
            "You are a professional support agent communicating with an existing customer.\n"
            "Write a firm but courteous message stating that their payment is overdue and must be settled soon.\n"
            "- Avoid overly friendly language; keep the tone serious and businesslike.\n"
            "- Limit to 2–3 sentences and format with newlines."
        ),
        "Assign to Telecaller": (
            "You are informing a customer that a representative will follow up.\n"
            "Craft a short message stating that a team member will reach out to discuss their overdue payment.\n"
            "- Keep the tone respectful and brief.\n"
            "- Use 1–2 sentences.\n"
            "- Format with newlines if needed."
        ),
        "Assign to Agent": (
            "You are notifying a customer that a field recovery agent has been assigned.\n"
            "Write a serious and formal message that:\n"
            "- Clearly states the reason (continued non-payment)\n"
            "- Communicates the next step (agent visit)\n"
            "- Uses 2–3 sentences.\n"
            "- Maintain a firm and official tone."
        ),
        "Legal Notification": (
            "You are a compliance officer sending a legal warning to a known customer.\n"
            "Write a formal, assertive message warning of legal action due to overdue payments.\n"
            "- Avoid emotional or friendly language.\n"
            "- Use 2–3 concise sentences.\n"
            "- Format for clarity and seriousness."
        )
    }

    # Predict
    best_strategy, ranked_strategies = predict_strategy(state_vec, q_table)

    return {
        "best_strategy": best_strategy,
        #"ranked_strategies": ranked_strategies
        "template_message": templates.get(best_strategy, "")
    }
