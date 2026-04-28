import telebot
import os
from dotenv import load_dotenv

from main import run_chatbot, extract_all_symptoms
from utils.db import save_user_query

# ----------------------------
# Load environment
# ----------------------------
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not TOKEN:
    raise ValueError("❌ TELEGRAM_BOT_TOKEN not found")

bot = telebot.TeleBot(TOKEN)


# ----------------------------
# Start Command
# ----------------------------
@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(
        message.chat.id,
        "👋 Hello! I'm your AI Health Assistant 🤖\n\n"
        "Just type your symptoms:\n"
        "• fever and cough\n"
        "• jwaram and talanoppi\n"
        "• bukhar aur khansi"
    )


# ----------------------------
# MAIN HANDLER
# ----------------------------
@bot.message_handler(func=lambda message: True)
def handle_message(message):

    user_input = message.text.strip()

    if not user_input:
        bot.send_message(message.chat.id, "❌ Please enter symptoms.")
        return

    user_id = message.from_user.id
    username = message.from_user.username or "NoUsername"
    full_name = message.from_user.first_name or "Unknown"

    try:
        print("\n==============================")
        print("📩 INPUT:", user_input)

        # ----------------------------
        # STEP 1: AI PIPELINE (MAIN LOGIC)
        # ----------------------------
        response = run_chatbot(user_input)

        if not response:
            response = "😕 Unable to process symptoms. Try again."

        print("📤 RESPONSE GENERATED")

        # ----------------------------
        # STEP 2: Send response immediately
        # ----------------------------
        bot.send_chat_action(message.chat.id, "typing")
        bot.send_message(message.chat.id, response)

        # ----------------------------
        # STEP 3: Extract symptoms (after response)
        # ----------------------------
        symptoms = extract_all_symptoms(user_input)

        print("🧾 SYMPTOMS:", symptoms)

        # ----------------------------
        # STEP 4: Save to DB safely
        # ----------------------------
        try:
            save_user_query(
                user_id=user_id,
                username=username,
                full_name=full_name,
                symptoms=symptoms,
                response=response
            )
            print("💾 DB SAVED SUCCESSFULLY")

        except Exception as db_error:
            print("❌ DB ERROR (ignored):", db_error)

        print("==============================\n")

    except Exception as e:
        print("❌ BOT ERROR:", e)

        bot.send_message(
            message.chat.id,
            "⚠️ Server error occurred. Please try again."
        )


# ----------------------------
# RUN BOT
# ----------------------------
print("🤖 Telegram Bot Running...")
bot.infinity_polling(skip_pending=True)