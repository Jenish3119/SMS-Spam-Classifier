import os
import sqlite3
from datetime import datetime

import joblib
import streamlit as st


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩",
    layout="centered",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "spam_classifier_optimized.pkl")
DB_PATH = os.path.join(BASE_DIR, "spam_blocklist.db")
HAM_WC_PATH = os.path.join(BASE_DIR, "ham_wc.png")
SPAM_WC_PATH = os.path.join(BASE_DIR, "spam_wc.png")


# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_artifact():
    artifact = joblib.load(MODEL_PATH)
    return (
        artifact["pipeline"],
        float(artifact["threshold"]),
        artifact["calibrated_model"],
    )


model, threshold, calibrated_model = load_artifact()


# -----------------------------
# Database
# -----------------------------
def get_connection():
    return sqlite3.connect(DB_PATH)


def init_database():
    with get_connection() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS blocked_senders (
                sender TEXT PRIMARY KEY,
                blocked_at TEXT NOT NULL
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT NOT NULL,
                message TEXT NOT NULL,
                prediction TEXT NOT NULL,
                spam_probability REAL,
                checked_at TEXT NOT NULL,
                feedback TEXT
            )
            """
        )

        conn.commit()


def is_sender_blocked(sender):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM blocked_senders WHERE sender = ?",
            (sender,),
        )
        return cur.fetchone() is not None


def block_sender(sender):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO blocked_senders (sender, blocked_at)
            VALUES (?, ?)
            """,
            (
                sender,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()


def unblock_sender(sender):
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM blocked_senders WHERE sender = ?",
            (sender,),
        )
        conn.commit()


def get_blocked_senders():
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT sender, blocked_at
            FROM blocked_senders
            ORDER BY blocked_at DESC
            """
        ).fetchall()


def save_prediction(sender, message, prediction, probability):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO prediction_history
            (sender, message, prediction, spam_probability, checked_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                sender,
                message,
                prediction,
                float(probability),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()


def get_history():
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT
                id,
                sender,
                message,
                prediction,
                spam_probability,
                checked_at,
                feedback
            FROM prediction_history
            ORDER BY id DESC
            """
        ).fetchall()


def save_feedback(prediction_id, feedback):
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE prediction_history
            SET feedback = ?
            WHERE id = ?
            """,
            (feedback, prediction_id),
        )
        conn.commit()


init_database()


# -----------------------------
# Session state
# -----------------------------
if "result" not in st.session_state:
    st.session_state.result = None

if "sender" not in st.session_state:
    st.session_state.sender = None

if "probability" not in st.session_state:
    st.session_state.probability = None


def clear_result():
    st.session_state.result = None
    st.session_state.sender = None
    st.session_state.probability = None


def rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# -----------------------------
# Navigation
# -----------------------------
st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Try the Model",
        "Prediction History",
        "Blocklist",
        "Dataset Insights",
        "About",
    ],
)


# -----------------------------
# Home
# -----------------------------
if page == "Home":
    st.title("📩 SMS Spam Detection System")

    st.write(
        "A simple end-to-end SMS spam filtering project using "
        "character-level TF-IDF and a tuned Linear SVM."
    )

    st.markdown(
        """
        ### Features
        - Spam / Ham prediction
        - Calibrated spam-risk score
        - Sender block / unblock
        - Persistent SQLite blocklist
        - Prediction history
        - Correct / Incorrect feedback
        """
    )


# -----------------------------
# Try model
# -----------------------------
elif page == "Try the Model":
    st.title("🧪 Try the Model")

    sender = st.text_input(
        "Sender ID / Phone Number",
        placeholder="Example: AD-OFFER",
    )

    message = st.text_area(
        "SMS Message",
        placeholder="Enter a message here...",
        height=150,
    )

    if st.button("Check Message"):
        clean_sender = sender.strip().upper()
        clean_message = message.strip()

        if not clean_sender:
            st.warning("Please enter a sender.")

        elif not clean_message:
            st.warning("Please enter a message.")

        elif is_sender_blocked(clean_sender):
            st.session_state.result = "BLOCKED"
            st.session_state.sender = clean_sender
            st.session_state.probability = None

        else:
            score = float(
                model.decision_function([clean_message])[0]
            )

            prediction = 1 if score >= threshold else 0

            probability = float(
                calibrated_model.predict_proba([clean_message])[0][1]
            )

            label = "SPAM" if prediction == 1 else "HAM"

            save_prediction(
                clean_sender,
                clean_message,
                label,
                probability,
            )

            st.session_state.result = label
            st.session_state.sender = clean_sender
            st.session_state.probability = probability

    result = st.session_state.result

    if result == "BLOCKED":
        st.error(
            f"🚫 {st.session_state.sender} is already blocked."
        )

    elif result == "SPAM":
        probability = st.session_state.probability * 100

        st.error("🚨 SPAM MESSAGE DETECTED")
        st.metric("Spam Risk Score", f"{probability:.2f}%")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🚫 Block Sender"):
                block_sender(st.session_state.sender)
                clear_result()
                rerun()

        with col2:
            if st.button("✅ Allow Sender"):
                clear_result()
                rerun()

    elif result == "HAM":
        probability = st.session_state.probability * 100

        st.success("✅ HAM MESSAGE")
        st.metric("Spam Risk Score", f"{probability:.2f}%")


# -----------------------------
# Prediction history
# -----------------------------
elif page == "Prediction History":
    st.title("📜 Prediction History")

    history = get_history()

    if not history:
        st.info("No predictions yet.")

    for row in history:
        (
            prediction_id,
            sender,
            message,
            prediction,
            probability,
            checked_at,
            feedback,
        ) = row

        with st.expander(
            f"{prediction} | {sender} | {checked_at}"
        ):
            st.write(f"**Sender:** {sender}")
            st.write(f"**Message:** {message}")
            st.write(f"**Prediction:** {prediction}")
            st.write(
                f"**Spam Risk:** {float(probability) * 100:.2f}%"
            )

            if feedback:
                st.write(f"**Feedback:** {feedback}")

            else:
                col1, col2 = st.columns(2)

                with col1:
                    if st.button(
                        "👍 Correct",
                        key=f"correct_{prediction_id}",
                    ):
                        save_feedback(
                            prediction_id,
                            "CORRECT",
                        )
                        rerun()

                with col2:
                    if st.button(
                        "👎 Incorrect",
                        key=f"incorrect_{prediction_id}",
                    ):
                        save_feedback(
                            prediction_id,
                            "INCORRECT",
                        )
                        rerun()


# -----------------------------
# Blocklist
# -----------------------------
elif page == "Blocklist":
    st.title("🚫 Blocked Senders")

    blocked = get_blocked_senders()

    if not blocked:
        st.info("No blocked senders.")

    for sender, blocked_at in blocked:
        col1, col2, col3 = st.columns([3, 3, 2])

        col1.write(sender)
        col2.write(blocked_at)

        if col3.button(
            "Unblock",
            key=f"unblock_{sender}",
        ):
            unblock_sender(sender)
            rerun()


# -----------------------------
# Dataset insights
# -----------------------------
elif page == "Dataset Insights":
    st.title("📊 Dataset Insights")

    if os.path.exists(HAM_WC_PATH):
        st.subheader("Ham WordCloud")
        st.image(HAM_WC_PATH)

    if os.path.exists(SPAM_WC_PATH):
        st.subheader("Spam WordCloud")
        st.image(SPAM_WC_PATH)

    st.write(
        "Character-level TF-IDF helps capture spelling variations, "
        "numbers, short codes and noisy SMS patterns."
    )


# -----------------------------
# About
# -----------------------------
elif page == "About":
    st.title("ℹ️ About")

    st.markdown(
        """
        **Pipeline:**  
        Raw SMS → Character-level TF-IDF → Tuned Linear SVM →
        Optimized Decision Threshold → Spam / Ham

        A calibrated classifier is used to produce the Spam Risk Score.

        **Technologies:** Python, Scikit-learn, Streamlit, SQLite, Joblib
        """
    )
