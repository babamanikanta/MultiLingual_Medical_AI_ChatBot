import telebot
import os
import time
from dotenv import load_dotenv

# ✅ UPDATED IMPORTS
from main import run_chatbot, extract_all_symptoms
from utils.db import save_user_query

# ----------------------------
# Load environment
# ----------------------------
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not TOKEN:
    raise ValueError("❌ TELEGRAM_BOT_TOKEN not found in .env file")

bot = telebot.TeleBot(TOKEN)

# ----------------------------
# Small Talk Handler
# ----------------------------
def handle_small_talk(text):
    text = text.lower()

    greetings = ["hi", "hello", "hey", "hii", "namaste"]
    thanks = ["thanks", "thank you", "thx"]

    if any(word in text for word in greetings):
        return (
            "👋 Hello! I'm your AI Health Assistant 🤖\n\n"
            "Tell me your symptoms like:\n"
            "• fever and cough\n"
            "• headache and vomiting\n"
        )

    if any(word in text for word in thanks):
        return "😊 You're welcome! Stay healthy."

    return None


# ----------------------------
# Start Command
# ----------------------------
@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(
        message.chat.id,
        f"""👋 Hello {message.from_user.first_name}!

🧠 I'm your AI Health Assistant 🤖

I understand:
🌍 English | Hindi | Telugu | Hinglish

💬 Try:
• fever and headache
• bukhar aur khansi
• jwaram and tala noppi

⚠️ Disclaimer:
This is an AI assistant, not a medical doctor.
Always consult a healthcare professional for serious conditions.
"""
    )


# ----------------------------
# Handle Messages
# ----------------------------
@bot.message_handler(func=lambda message: True)
def handle_message(message):

    user_input = message.text.strip()
    user_id = message.from_user.id
    user_name = message.from_user.username or "NoUsername"
    full_name = message.from_user.first_name or "Unknown"

    if not user_input:
        bot.send_message(message.chat.id, "❌ Please enter valid symptoms.")
        return

    try:
        # ----------------------------
        # Small Talk First
        # ----------------------------
        small_talk = handle_small_talk(user_input)
        if small_talk:
            bot.send_message(message.chat.id, small_talk)
            return

        # ----------------------------
        # Typing effect
        # ----------------------------
        bot.send_chat_action(message.chat.id, "typing")
        time.sleep(0.7)

        # ----------------------------
        # AI response (CORRECT PIPELINE)
        # ----------------------------
        response = run_chatbot(user_input)

        if not response:
            response = (
                "😕 I couldn't understand your symptoms.\n\n"
                "Try like:\n"
                "• fever and cough\n"
                "• headache and vomiting"
            )

        # ----------------------------
        # Add Disclaimer
        # ----------------------------
        response += (
            "\n\n⚠️ Note:\n"
            "This is AI-generated advice and not a medical diagnosis.\n"
            "Please consult a doctor if symptoms persist."
        )

        # ----------------------------
        # ✅ FIXED SYMPTOM EXTRACTION
        # ----------------------------
        symptoms = extract_all_symptoms(user_input)

        # Debug (optional)
        print("INPUT:", user_input)
        print("SYMPTOMS:", symptoms)

        # ----------------------------
        # Save to DB
        # ----------------------------
        save_user_query(
            user_id=user_id,
            username=user_name,
            full_name=full_name,
            symptoms=symptoms,
            response=response
        )

        # ----------------------------
        # Send response
        # ----------------------------
        bot.send_message(message.chat.id, response)

    except Exception as e:
        print("Bot Error:", e)

        bot.send_message(
            message.chat.id,
            "⚠️ Server error occurred. Please try again."
        )


# ----------------------------
# Run Bot
# ----------------------------
print("🤖 Telegram Bot Running...")
bot.infinity_polling(timeout=10, long_polling_timeout=5)