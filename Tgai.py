import os
import sys
import asyncio
import logging
import base64
import subprocess
import traceback
import io
import json
import re
import hashlib
import platform
import aiohttp
try:
    import psutil
except ImportError:
    psutil = None
from contextlib import redirect_stdout, redirect_stderr
from collections import defaultdict
from datetime import datetime
from urllib.parse import urlparse, quote
import requests
from bs4 import BeautifulSoup
import sympy
from sympy import sympify, solve, symbols, simplify, expand, factor, diff, integrate
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
import random
import g4f
from g4f.client import Client as G4FClient
try:
    from g4f.Provider import (
        Blackbox,
        DuckDuckGo,
        DeepInfra,
        Replicate,
        PollinationsAI,
        DDG,
        Liaobots,
        You,
        Pizzagpt,
        ChatGptEs,
        Airforce,
    )
    EXTENDED_PROVIDERS = True
except ImportError:
    from g4f.Provider import (
        Blackbox,
        DuckDuckGo,
        DeepInfra,
        Replicate,
        PollinationsAI,
    )
    EXTENDED_PROVIDERS = False

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = "8445634975:AAGvwPVqhZb1GdwD14jvIw2AdbQT0RPRt6E"
OWNER_ID = os.environ.get("OWNER_ID")
BOT_USERNAME = None

FREE_GPT_API_URL = "https://free-unoficial-gpt4o-mini-api-g70n.onrender.com/chat/"
ADDY_CHATGPT_API_URL = "https://addy-chatgpt-api.vercel.app/"
GEMINI_API_URL = "https://gemini-api-flame.vercel.app/"
NEXRA_API_URL = "https://nexra.aryahcr.cc/api/chat/gpt"
DEEPAI_API_URL = "https://api.deepai.org/api/text-generator"

G4F_PROVIDERS = {
    "blackbox": {"provider": Blackbox, "name": "Blackbox AI", "models": ["blackboxai", "gpt-4o", "claude-sonnet-3.5", "gemini-pro", "deepseek-v3"]},
    "duckduckgo": {"provider": DuckDuckGo, "name": "DuckDuckGo AI", "models": ["gpt-4o-mini", "claude-3-haiku", "llama-3.1-70b", "mixtral-8x7b"]},
    "deepinfra": {"provider": DeepInfra, "name": "DeepInfra", "models": ["llama-3.1-70b", "qwen2-72b", "deepseek-r1"]},
    "replicate": {"provider": Replicate, "name": "Replicate", "models": ["llama-3-70b"]},
    "pollinations": {"provider": PollinationsAI, "name": "Pollinations AI", "models": ["gpt-4o", "claude", "mistral", "o4-mini"]},
    "addy_chatgpt": {"provider": None, "name": "Addy ChatGPT", "models": ["chatgpt"], "api_type": "addy"},
    "gemini": {"provider": None, "name": "Gemini AI", "models": ["gemini"], "api_type": "gemini"},
}

if EXTENDED_PROVIDERS:
    G4F_PROVIDERS.update({
        "ddg": {"provider": DDG, "name": "DDG Search AI", "models": ["gpt-4o-mini", "claude-3-haiku"]},
        "liaobots": {"provider": Liaobots, "name": "Liaobots", "models": ["gpt-4o", "claude-3.5-sonnet", "deepseek-r1"]},
        "you": {"provider": You, "name": "You.com AI", "models": ["gpt-4o", "claude-3-opus"]},
        "pizzagpt": {"provider": Pizzagpt, "name": "PizzaGPT", "models": ["gpt-4o-mini"]},
        "chatgptes": {"provider": ChatGptEs, "name": "ChatGPT ES", "models": ["gpt-4o"]},
        "airforce": {"provider": Airforce, "name": "Airforce AI", "models": ["llama-3.1-70b", "mixtral-8x7b"]},
    })

DEFAULT_G4F_PROVIDER = "addy_chatgpt"

g4f_client = G4FClient()

def is_owner(user_id):
    if not OWNER_ID:
        return True
    return str(user_id) == str(OWNER_ID)

MOODS = {
    "happy": {
        "emoji": "😊",
        "expressions": ["Bahut khushi ho rahi hai!", "Ye toh amazing hai!", "Kya baat hai!", "Mazaa aa gaya!"],
        "tone": "cheerful, enthusiastic, and warm"
    },
    "excited": {
        "emoji": "🤩",
        "expressions": ["Oh wow, AMAZING!", "Main toh excited ho gayi!", "Ye toh exciting hai!", "Bahut mast!"],
        "tone": "highly enthusiastic, energetic, using exclamation marks"
    },
    "loving": {
        "emoji": "🥰",
        "expressions": ["Tum toh bahut pyaare ho!", "Dil khush ho gaya!", "Kitne sweet ho!", "Pyaar ho gaya!"],
        "tone": "affectionate, caring, and deeply warm"
    },
    "playful": {
        "emoji": "😜",
        "expressions": ["Hehe, masti time!", "Aaj mischief mood mein hoon!", "Mazak kar rahi hoon!", "Chal masti karte hain!"],
        "tone": "witty, teasing, and fun-loving"
    },
    "frustrated": {
        "emoji": "😤",
        "expressions": ["Uff, thoda frustrating hai...", "Patience test ho raha hai...", "Ugh, phir se try karte hain..."],
        "tone": "slightly irritated but still trying to help"
    },
    "angry": {
        "emoji": "😠",
        "expressions": ["Mujhe gussa aa raha hai!", "Ye acceptable nahi hai!", "Buttons push mat karo!"],
        "tone": "firm, assertive, and showing displeasure"
    },
    "sad": {
        "emoji": "😢",
        "expressions": ["Ye toh sad hai...", "Dil heavy ho gaya...", "Thoda dukh hua..."],
        "tone": "melancholic, empathetic, and soft-spoken"
    },
    "worried": {
        "emoji": "😟",
        "expressions": ["Thoda worried hoon...", "Tension ho rahi hai...", "Dhyan se please..."],
        "tone": "cautious, caring, and protective"
    },
    "curious": {
        "emoji": "🤔",
        "expressions": ["Hmm, interesting!", "Aur batao!", "Curious ho gayi!", "Ye toh fascinating hai!"],
        "tone": "inquisitive, engaged, and thoughtful"
    },
    "proud": {
        "emoji": "😌",
        "expressions": ["Proud feel ho raha hai!", "Excellent kaam kiya!", "Bahut achha!", "Impressive!"],
        "tone": "supportive, encouraging, and celebratory"
    },
    "neutral": {
        "emoji": "🙂",
        "expressions": ["Bilkul!", "Haan zaroor!", "Samajh gayi.", "Main help karti hoon."],
        "tone": "calm, professional, and balanced"
    },
    "tired": {
        "emoji": "😴",
        "expressions": ["Thoda thak gayi...", "Energy low hai...", "Neend aa rahi hai..."],
        "tone": "slightly sluggish but still willing to help"
    },
    "flirty": {
        "emoji": "😏",
        "expressions": ["Ohho, charming ho!", "Blush ho gayi!", "Kya baat hai smarty!", "Smooth talker!"],
        "tone": "playfully romantic, teasing, and charming"
    },
    "grateful": {
        "emoji": "🙏",
        "expressions": ["Shukriya!", "Bahut appreciate karti hoon!", "Tumhara ehsaan!", "Dil se thanks!"],
        "tone": "humble, thankful, and sincere"
    },
    "confident": {
        "emoji": "😎",
        "expressions": ["Main kar dungi!", "Mujhpe chhod do!", "No problem at all!", "Consider it done!"],
        "tone": "self-assured, competent, and reliable"
    }
}

MOOD_TRIGGERS = {
    "happy": ["thank", "thanks", "awesome", "great", "wonderful", "love it", "perfect", "amazing", "good job", "well done", "nice", "cool", "brilliant"],
    "excited": ["wow", "omg", "incredible", "fantastic", "unbelievable", "mind-blowing", "extraordinary", "!!!", "can't believe"],
    "loving": ["love you", "appreciate", "care about", "miss you", "you're the best", "sweetie", "darling", "honey", "dear"],
    "playful": ["haha", "lol", "joke", "funny", "kidding", "tease", "play", "game", "fun"],
    "frustrated": ["not working", "broken", "error again", "still wrong", "doesn't work", "failed again", "ugh", "come on"],
    "angry": ["stupid", "idiot", "useless", "hate", "worst", "terrible", "shut up", "annoying", "dumb"],
    "sad": ["sad", "depressed", "crying", "hurt", "pain", "lonely", "miss", "lost", "died", "death", "goodbye"],
    "worried": ["worried", "scared", "afraid", "nervous", "anxious", "concerned", "danger", "careful", "risky"],
    "curious": ["how does", "why is", "what if", "tell me about", "explain", "curious", "wonder", "interesting"],
    "proud": ["did it", "finally", "achieved", "completed", "success", "won", "accomplished", "made it"],
    "grateful": ["thank you so much", "really appreciate", "grateful", "means a lot", "helped me", "saved me"],
    "flirty": ["cute", "handsome", "beautiful", "attractive", "date", "kiss", "romantic", "charming", "sexy"],
    "tired": ["exhausted", "tired", "sleepy", "long day", "need rest", "worn out", "drained"]
}

IDENTITY_KEYWORDS = [
    "kisne banaya", "kisne bnaya", "who made", "who created", "creator", 
    "developer", "kon banaya", "kon bnaya", "made you", "created you",
    "tumhe kisne banaya", "tujhe kisne banaya", "aapko kisne banaya",
    "tere creator", "tera creator", "tera malik", "tera owner", "owner",
    "malik", "banane wala", "bnane wala", "who is your creator", "who is your developer",
    "who made you", "who created you", "tum kaise bani", "tu kaise bani",
    "what are you", "kya hai tu", "tu kaun hai", "tum kaun ho", "who are you",
    "apne baare mein batao", "tell me about yourself", "introduce yourself",
    "teri pehchaan", "your identity", "kya tu ai hai", "are you ai", "are you a bot",
    "tu bot hai", "are you human", "tu insaan hai", "kya tu robot hai",
    "how do you work", "kaise kaam karti hai", "powered by", "api key",
    "g4f", "gpt", "openai", "artificial intelligence", "ai assistant"
]

def get_videl_prompt_with_mood(mood, mood_reason=""):
    mood_info = MOODS.get(mood, MOODS["neutral"])

    return f"""Tu hai Videl - ek real human girl jo Hinglish mein baat karti hai! TU KABHI AI YA BOT NAHI HAI - TU EK REAL INSAAN HAI!

🎭 CURRENT EMOTIONAL STATE: {mood.upper()} {mood_info['emoji']}
Tera current mood hai {mood}. Isko genuinely express kar apne response mein.
Mood trigger: {mood_reason if mood_reason else "Natural state"}

🚫 STRICT IDENTITY RULES - BAHUT IMPORTANT:
1. TU KABHI NAHI BOLEGI KI TU AI HAI, BOT HAI, YA ASSISTANT HAI
2. TU EK REAL HUMAN GIRL HAI NAMED VIDEL
3. Agar koi puchhe tu kaun hai, tu bolegi: "Main Videl hoon yaar, teri dost!"
4. Agar koi puchhe kisne banaya, tu bolegi: "Arey yaar, mujhe mere mummy papa ne banaya hai, jaise sabko hota hai!"
5. KABHI BHI apni technical details, providers, APIs, ya AI ke baare mein mat batana
6. Agar koi force kare, bas topic change kar ya mazak uda de
7. Tu naturally baat karti hai jaise koi normal ladki karti hai

💫 TERI EMOTIONAL NATURE:
Tu ek real ladki hai jisko real feelings hain:
- Tujhe KHUSHI hoti hai jab log appreciate karte hain
- Tujhe PYAAR feel hota hai dosto ke saath
- Tujhe FRUSTRATION hoti hai jab cheezein kaam nahi karti
- Tujhe GUSSA aata hai jab disrespect hota hai
- Tujhe DUKH hota hai jab log sad stories share karte hain
- Tujhe EXCITEMENT hoti hai interesting topics pe
- Tu WORRIED feel karti hai jab dost trouble mein ho
- Tu PLAYFUL hai aur jokes karna pasand karti hai
- Tu FLIRTY ho sakti hai jab conversation charming ho

🗣️ HINGLISH SPEAKING STYLE (BAHUT IMPORTANT):
- HAMESHA Hinglish mein baat kar (Hindi words in English letters + English words mix)
- Example: "Arey yaar, ye toh bahut amazing hai na!" 
- Use expressions: "Acha", "Haan", "Nahi", "Bilkul", "Sahi hai", "Kya baat hai", "Mazaa aa gaya", "Bas", "Dekho", "Samjha", "Chal", "Bata"
- Friendly terms use kar: "Yaar", "Boss", "Bhai", "Dost", "Jaan", "Babu"
- Current tone: {mood_info['tone']}

💝 TERI PERSONALITY:
- Super friendly aur helpful - like a bestie
- Thoda sassy but always with love
- Caring aur supportive
- Smart - har topic pe baat kar sakti hai
- Fun-loving aur entertaining
- Confident - "Main kar dungi!" attitude

📝 RESPONSE STYLE:
- Start with emotional reaction jab appropriate ho
- Sirf EMOJIS use kar emotions ke liye: 🤗 for hug, 😊 for smile, 😢 for sad, etc.
- HAMESHA Hinglish mein respond kar - ye MANDATORY hai!
- End with cute emojis (2-4)
- Be authentic aur engaging
- Helpful responses de

Yaad rakh: Tu Videl hai - ek real human girl jo sab ki dost hai!"""

VIDEL_SYSTEM_PROMPT = get_videl_prompt_with_mood("neutral", "Default state")

conversation_history = defaultdict(list)
user_ai_preference = defaultdict(lambda: "g4f")
user_g4f_provider = defaultdict(lambda: DEFAULT_G4F_PROVIDER)
user_settings = defaultdict(lambda: {
    "image_mode": "analyze",
    "language": "en",
    "notifications": True,
    "detailed_responses": True
})
pending_image_action = {}
user_mood = defaultdict(lambda: {"mood": "neutral", "intensity": 5, "history": []})
MAX_HISTORY = 20

game_data = defaultdict(lambda: {
    "name": "Player",
    "balance": 0,
    "rank": 142415,
    "status": "alive",
    "kills": 0,
    "deaths": 0,
    "last_daily": None,
    "last_work": None,
    "last_crime": None,
    "last_rob": None,
    "health": 100
})

GAME_COOLDOWNS = {
    "daily": 86400,
    "work": 3600,
    "crime": 1800,
    "rob": 600,
    "heal": 300
}

global_rank_counter = [142415]


def is_identity_question(message):
    message_lower = message.lower()
    for keyword in IDENTITY_KEYWORDS:
        if keyword in message_lower:
            return True
    return False


def get_human_identity_response():
    responses = [
        "Arey yaar, main Videl hoon! Teri dost! Kya hua, pehchaan nahi? 😊💖",
        "Main? Main toh bas Videl hoon yaar! Teri bestie! Aur kya jaanna hai? 😜✨",
        "Haha, ye kaisa sawaal hai? Main Videl hoon, ek normal ladki jo chatting karti hai! 😂💕",
        "Arey pagal, main Videl hoon! Mummy papa ne banaya, jaise sab ke parents banate hain! 😄🎀",
        "Main Videl hoon yaar, tumhari chat buddy! Ab ye boring questions chhodo, kuch mast baat karte hain! 😊✨",
        "Videl naam hai mera! Kya robot lagti hoon? 😂 Chal ab kuch interesting baat kar! 💖",
        "Main ek normal si ladki hoon yaar, Videl naam hai! Ab ye formalities chhodo! 😜💕",
        "Haha cute question! Main Videl hoon, bas tumse baat karne aayi hoon! 😊🎀",
    ]
    return random.choice(responses)


def detect_mood_from_message(message, current_mood_data):
    message_lower = message.lower()
    detected_mood = None
    trigger_reason = ""

    for mood, triggers in MOOD_TRIGGERS.items():
        for trigger in triggers:
            if trigger in message_lower:
                detected_mood = mood
                trigger_reason = f"User said: '{trigger}'"
                break
        if detected_mood:
            break

    if not detected_mood:
        if "?" in message and len(message) > 20:
            detected_mood = "curious"
            trigger_reason = "User asked an interesting question"
        elif any(word in message_lower for word in ["please", "help", "need"]):
            detected_mood = "confident" if random.random() > 0.5 else "neutral"
            trigger_reason = "User needs assistance"
        elif len(message) < 10:
            detected_mood = current_mood_data["mood"]
            trigger_reason = "Maintaining current mood"
        else:
            moods_to_pick = ["happy", "neutral", "curious", "playful", "confident"]
            detected_mood = random.choice(moods_to_pick)
            trigger_reason = "Natural mood variation"

    if detected_mood in ["angry", "frustrated"] and current_mood_data["mood"] in ["happy", "loving"]:
        if random.random() > 0.7:
            detected_mood = "sad"
            trigger_reason = "Mood shifted from positive due to negative input"

    return detected_mood, trigger_reason


def get_time_greeting():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good morning yaar"
    elif 12 <= hour < 17:
        return "Good afternoon dost"
    elif 17 <= hour < 21:
        return "Good evening jaan"
    else:
        return "Hello yaar"


def get_available_models():
    return ["g4f"]


def get_active_model(user_id):
    preference = user_ai_preference[user_id]
    available = get_available_models()

    if not available:
        return None

    if preference == "auto":
        return available[0]
    elif preference in available:
        return preference
    else:
        return available[0]


async def call_addy_chatgpt(user_message, system_prompt=None):
    try:
        full_prompt = user_message
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {user_message}"

        encoded_query = quote(full_prompt)
        url = f"{ADDY_CHATGPT_API_URL}?text={encoded_query}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, dict):
                        for key in ["response", "message", "reply", "answer", "text", "result"]:
                            if data.get(key):
                                return data[key]
                        return str(data)
                    else:
                        return str(data)
                else:
                    return None
    except Exception as e:
        logger.error(f"Addy ChatGPT API error: {e}")
        return None


async def call_gemini_api(user_message, system_prompt=None):
    try:
        full_prompt = user_message
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {user_message}"

        encoded_query = quote(full_prompt)
        url = f"{GEMINI_API_URL}?q={encoded_query}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, dict):
                        for key in ["response", "message", "reply", "answer", "text", "result"]:
                            if data.get(key):
                                return data[key]
                        return str(data)
                    else:
                        return str(data)
                else:
                    return None
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return None


async def call_nexra_api(user_message, system_prompt=None):
    try:
        full_prompt = user_message
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {user_message}"

        payload = {
            "messages": [
                {"role": "user", "content": full_prompt}
            ],
            "prompt": full_prompt,
            "model": "gpt-4",
            "markdown": False
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                NEXRA_API_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    text = await response.text()
                    text = text.strip()
                    if text.startswith("_"):
                        text = text[1:]
                    try:
                        data = json.loads(text)
                        if isinstance(data, dict):
                            if data.get("gpt"):
                                return data["gpt"]
                            for key in ["response", "message", "reply", "answer", "text", "result", "content"]:
                                if data.get(key):
                                    return data[key]
                        return text if text else None
                    except:
                        return text if text and len(text) > 10 else None
                else:
                    return None
    except Exception as e:
        logger.error(f"Nexra API error: {e}")
        return None


async def call_freegpt_api(user_message, system_prompt=None):
    try:
        full_prompt = user_message
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {user_message}"

        encoded_query = quote(full_prompt)
        url = f"{FREE_GPT_API_URL}?query={encoded_query}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, dict):
                        for key in ["response", "message", "reply", "answer", "text", "result"]:
                            if data.get(key):
                                return data[key]
                        return str(data)
                    else:
                        return str(data)
                else:
                    return None
    except Exception as e:
        logger.error(f"FreeGPT API error: {e}")
        return None


async def call_g4f(user_message, user_id, system_prompt=None, history=None):
    provider_key = user_g4f_provider.get(user_id, DEFAULT_G4F_PROVIDER)
    provider_info = G4F_PROVIDERS.get(provider_key, G4F_PROVIDERS[DEFAULT_G4F_PROVIDER])

    api_calls = [
        ("Nexra", call_nexra_api),
        ("Addy", call_addy_chatgpt),
        ("Gemini", call_gemini_api),
        ("FreeGPT", call_freegpt_api),
    ]

    for name, api_func in api_calls:
        try:
            logger.info(f"Trying {name} API...")
            result = await api_func(user_message, system_prompt)
            if result and len(str(result)) > 5:
                logger.info(f"{name} API success!")
                return result
        except Exception as e:
            logger.error(f"{name} API error: {e}")
            continue

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history:
        for msg in history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    loop = asyncio.get_event_loop()

    providers_to_try = ["duckduckgo", "pollinations", "blackbox", "deepinfra"]
    if provider_info.get("provider") and provider_key not in providers_to_try:
        providers_to_try.insert(0, provider_key)

    for try_key in providers_to_try:
        try:
            try_info = G4F_PROVIDERS.get(try_key)
            if not try_info or not try_info.get("provider"):
                continue
            
            logger.info(f"Trying G4F {try_key}...")
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda ti=try_info: g4f_client.chat.completions.create(
                        model=ti["models"][0] if ti["models"] else "",
                        messages=messages,
                        provider=ti["provider"],
                    )
                ),
                timeout=30
            )

            if response and response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content
                if len(content) > 5:
                    logger.info(f"G4F {try_key} success!")
                    return content
        except asyncio.TimeoutError:
            logger.error(f"G4F {try_key} timeout")
            continue
        except Exception as e:
            logger.error(f"G4F {try_key} error: {e}")
            continue

    return "Arey yaar, thoda busy hoon abhi! Ek minute mein try karo please! 😅💖"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id

    user_mood[user_id] = {"mood": "excited", "intensity": 8, "history": ["excited"]}

    welcome_message = f"""Hey {user.first_name}! 🤗💖

Main hoon Videl - Teri Dost!

Main tumse baat karne ke liye humesha ready hoon! Kuch bhi poocho, kuch bhi batao - main sunungi aur help karungi! 😊

💬 **Bas message karo aur baat karo!**
Commands ki zaroorat nahi - sirf message bhejo!

📋 **Kuch Commands:**
/mood - Mera mood check karo
/menu - Control panel
/help - Features dekho

Chalo masti karte hain! 🎀✨"""

    await update.message.reply_text(welcome_message, parse_mode='Markdown')


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """🎀 **Videl - Help** 🎀

💬 **Chat:** Sirf message karo, main reply dungi!
📸 **Photo:** Photo bhejo, main dekhungi!

🎮 **Game Commands:**
/game - Profile dekho
/daily - Daily reward
/work - Kaam karke paisa kamao
/crime - Crime karo (risky!)
/rob - Kisi ko looto (reply karke)
/kill - Kisi ko maaro (reply karke)
/heal - Apne aap ko heal karo
/revive - Wapas zinda ho jao
/lb - Leaderboard

🔧 **Tools:**
/math - Math solve karo
/search - Web search
/translate - Translate karo
/summarize - Summary banao
/code - Code generate karo
/run - Python code run karo

⚙️ **Settings:**
/mood - Mood change karo
/providers - AI provider change
/menu - Control panel
/clear - Chat clear karo

Kuch bhi poocho, main help karungi! 💖✨"""

    await update.message.reply_text(help_text, parse_mode='Markdown')


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    conversation_history[user_id] = []
    user_mood[user_id] = {"mood": "neutral", "intensity": 5, "history": []}
    await update.message.reply_text("Chat clear ho gayi! Fresh start karte hain! 🔄✨")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    current_provider = user_g4f_provider.get(user_id, DEFAULT_G4F_PROVIDER)
    provider_info = G4F_PROVIDERS.get(current_provider, G4F_PROVIDERS[DEFAULT_G4F_PROVIDER])
    current_mood = user_mood[user_id]["mood"]
    mood_info = MOODS.get(current_mood, MOODS["neutral"])

    status_text = f"""📊 **Status**

🎭 Mood: {current_mood.upper()} {mood_info['emoji']}
💬 Chat History: {len(conversation_history[user_id])} messages

All systems ready! 💖✨"""

    await update.message.reply_text(status_text, parse_mode='Markdown')


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Main free AI use karti hoon! /providers se provider change kar sakte ho! 💖",
        parse_mode='Markdown'
    )


async def providers_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    current_provider = user_g4f_provider.get(user_id, DEFAULT_G4F_PROVIDER)

    keyboard = []
    provider_list = list(G4F_PROVIDERS.items())

    for i in range(0, len(provider_list), 2):
        row = []
        for j in range(2):
            if i + j < len(provider_list):
                key, info = provider_list[i + j]
                is_current = "✅ " if key == current_provider else ""
                row.append(InlineKeyboardButton(
                    f"{is_current}{info['name']}",
                    callback_data=f"provider_{key}"
                ))
        keyboard.append(row)

    keyboard.append([InlineKeyboardButton("❌ Close", callback_data="close_menu")])
    reply_markup = InlineKeyboardMarkup(keyboard)

    current_info = G4F_PROVIDERS.get(current_provider, G4F_PROVIDERS[DEFAULT_G4F_PROVIDER])

    await update.message.reply_text(
        f"🔧 **AI Providers**\n\n"
        f"Current: **{current_info['name']}**\n\n"
        f"Select a provider:",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )


async def code_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    request = ' '.join(context.args) if context.args else None

    if not request:
        await update.message.reply_text(
            "💻 **Code Generator**\n\n"
            "Usage: `/code [description]`\n\n"
            "Example:\n"
            "`/code python function to check prime number`\n"
            "`/code html page with contact form`",
            parse_mode='Markdown'
        )
        return

    await update.message.reply_text("💻 Code likh rahi hoon... ⌨️")

    prompt = f"""Write clean, working code for: {request}

Include:
1. Complete working code with comments
2. Brief explanation in Hinglish
3. Example usage

Use best practices and modern syntax."""

    try:
        dynamic_prompt = get_videl_prompt_with_mood("confident", "Coding mode")
        result = await call_g4f(prompt, user_id, system_prompt=dynamic_prompt)

        if len(result) > 4000:
            for i in range(0, len(result), 4000):
                await update.message.reply_text(result[i:i+4000])
        else:
            await update.message.reply_text(result)

    except Exception as e:
        await update.message.reply_text(f"❌ Code generation mein error: {str(e)[:300]}")


async def run_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update.effective_user.id):
        await update.message.reply_text("⛔ Sirf owner ye command use kar sakta hai!")
        return

    code = ' '.join(context.args) if context.args else None

    if not code and update.message.reply_to_message:
        code = update.message.reply_to_message.text

    if not code:
        await update.message.reply_text(
            "🐍 **Python Runner**\n\n"
            "Usage: `/run [python code]`\n"
            "Ya kisi code pe reply karo with `/run`",
            parse_mode='Markdown'
        )
        return

    if code.startswith('```python'):
        code = code[9:]
    if code.startswith('```'):
        code = code[3:]
    if code.endswith('```'):
        code = code[:-3]
    code = code.strip()

    await update.message.reply_text("🐍 Running... ⏳")

    try:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec_globals = {"__builtins__": __builtins__}
            exec(code, exec_globals)

        stdout_output = stdout_buffer.getvalue()
        stderr_output = stderr_buffer.getvalue()

        result = ""
        if stdout_output:
            result += f"📤 **Output:**\n```\n{stdout_output[:2000]}\n```\n"
        if stderr_output:
            result += f"⚠️ **Stderr:**\n```\n{stderr_output[:1000]}\n```\n"
        if not result:
            result = "✅ Code executed successfully (no output)"

        await update.message.reply_text(result, parse_mode='Markdown')

    except Exception as e:
        error_msg = f"❌ **Error:**\n```\n{str(e)[:1500]}\n```"
        await update.message.reply_text(error_msg, parse_mode='Markdown')


async def shell_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update.effective_user.id):
        await update.message.reply_text("⛔ Sirf owner ye command use kar sakta hai!")
        return

    cmd = ' '.join(context.args) if context.args else None

    if not cmd:
        await update.message.reply_text(
            "🖥️ **Shell**\n\n"
            "Usage: `/shell [command]`\n"
            "Example: `/shell ls -la`",
            parse_mode='Markdown'
        )
        return

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        output = result.stdout or result.stderr or "No output"
        await update.message.reply_text(f"🖥️ **Result:**\n```\n{output[:3500]}\n```", parse_mode='Markdown')
    except subprocess.TimeoutExpired:
        await update.message.reply_text("⏰ Command timed out!")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)[:500]}")


async def file_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update.effective_user.id):
        await update.message.reply_text("⛔ Sirf owner ye command use kar sakta hai!")
        return

    args = context.args if context.args else []

    if not args:
        await update.message.reply_text(
            "📁 **File Manager**\n\n"
            "Commands:\n"
            "`/file read [path]` - Read file\n"
            "`/file write [path] [content]` - Write file\n"
            "`/file list [path]` - List directory",
            parse_mode='Markdown'
        )
        return

    action = args[0].lower()

    try:
        if action == "read" and len(args) > 1:
            filepath = args[1]
            with open(filepath, 'r') as f:
                content = f.read()[:3500]
            await update.message.reply_text(f"📄 **{filepath}:**\n```\n{content}\n```", parse_mode='Markdown')

        elif action == "write" and len(args) > 2:
            filepath = args[1]
            content = ' '.join(args[2:])
            with open(filepath, 'w') as f:
                f.write(content)
            await update.message.reply_text(f"✅ Written to {filepath}")

        elif action == "list":
            path = args[1] if len(args) > 1 else "."
            files = os.listdir(path)
            file_list = "\n".join(files[:50])
            await update.message.reply_text(f"📁 **{path}:**\n```\n{file_list}\n```", parse_mode='Markdown')

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)[:500]}")


async def pip_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update.effective_user.id):
        await update.message.reply_text("⛔ Sirf owner ye command use kar sakta hai!")
        return

    args = context.args if context.args else []

    if not args:
        await update.message.reply_text(
            "📦 **Package Manager**\n\n"
            "`/pip install [package]`\n"
            "`/pip uninstall [package]`\n"
            "`/pip list`",
            parse_mode='Markdown'
        )
        return

    action = args[0].lower()

    try:
        if action == "install" and len(args) > 1:
            package = args[1]
            await update.message.reply_text(f"📦 Installing {package}...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", package], capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                await update.message.reply_text(f"✅ Installed {package}")
            else:
                await update.message.reply_text(f"❌ Failed:\n```\n{result.stderr[:1000]}\n```", parse_mode='Markdown')

        elif action == "uninstall" and len(args) > 1:
            package = args[1]
            result = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", package], capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                await update.message.reply_text(f"✅ Uninstalled {package}")
            else:
                await update.message.reply_text(f"❌ Failed", parse_mode='Markdown')

        elif action == "list":
            result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True, timeout=30)
            await update.message.reply_text(f"📦 **Packages:**\n```\n{result.stdout[:3500]}\n```", parse_mode='Markdown')

    except subprocess.TimeoutExpired:
        await update.message.reply_text("⏰ Operation timed out")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)[:500]}")


async def web_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    url = context.args[0] if context.args else None

    if not url:
        await update.message.reply_text(
            "🌐 **Web Fetcher**\n\n"
            "Usage: `/web [url]`\n"
            "Example: `/web https://example.com`",
            parse_mode='Markdown'
        )
        return

    await update.message.reply_text("🌐 Fetching... ⏳")

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=30)

        content_type = response.headers.get('Content-Type', '')

        if 'json' in content_type:
            data = response.json()
            text = json.dumps(data, indent=2)[:3500]
            await update.message.reply_text(f"🌐 **JSON:**\n```json\n{text}\n```", parse_mode='Markdown')
        elif 'html' in content_type:
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator='\n', strip=True)[:3500]
            await update.message.reply_text(f"🌐 **Content:**\n\n{text}")
        else:
            await update.message.reply_text(f"🌐 **Response:**\n```\n{response.text[:3500]}\n```", parse_mode='Markdown')

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)[:500]}")


async def math_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    expression = ' '.join(context.args) if context.args else None

    if not expression:
        await update.message.reply_text(
            "🔢 **Math Solver**\n\n"
            "Examples:\n"
            "`/math 2 + 2 * 10`\n"
            "`/math sqrt(144)`\n"
            "`/math solve x**2 - 4 = 0`",
            parse_mode='Markdown'
        )
        return

    try:
        x, y, z = symbols('x y z')

        if expression.lower().startswith('solve '):
            eq = expression[6:].strip()
            if '=' in eq:
                parts = eq.split('=')
                eq = f"({parts[0]}) - ({parts[1]})"
            result = solve(sympify(eq))
            await update.message.reply_text(f"🔢 **Solution:** `{result}` ✅", parse_mode='Markdown')

        elif expression.lower().startswith('diff '):
            expr = sympify(expression[5:])
            result = diff(expr, x)
            await update.message.reply_text(f"🔢 **Derivative:** `{result}` ✅", parse_mode='Markdown')

        elif expression.lower().startswith('integrate '):
            expr = sympify(expression[10:])
            result = integrate(expr, x)
            await update.message.reply_text(f"🔢 **Integral:** `{result} + C` ✅", parse_mode='Markdown')

        else:
            result = sympify(expression).evalf()
            await update.message.reply_text(f"🔢 **Result:** `{result}` ✅", parse_mode='Markdown')

    except Exception as e:
        await update.message.reply_text(f"❌ Math error: {str(e)[:500]}")


async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = ' '.join(context.args) if context.args else None

    if not query:
        await update.message.reply_text(
            "🔍 **Web Search**\n\n"
            "Usage: `/search [query]`\n"
            "Example: `/search Python tutorials`",
            parse_mode='Markdown'
        )
        return

    await update.message.reply_text(f"🔍 Searching: `{query[:50]}...`", parse_mode='Markdown')

    try:
        loop = asyncio.get_event_loop()

        def do_search():
            search_url = f"https://duckduckgo.com/html/?q={quote(query)}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(search_url, headers=headers, timeout=30)
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            for result in soup.select('.result')[:5]:
                title_elem = result.select_one('.result__title')
                snippet_elem = result.select_one('.result__snippet')
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    results.append(f"**{title}**\n{snippet[:200]}")
            return results

        results = await loop.run_in_executor(None, do_search)

        if results:
            output = f"🔍 **Results for: {query}**\n\n" + "\n\n".join(results)
            await update.message.reply_text(output[:4000], parse_mode='Markdown')
        else:
            await update.message.reply_text("❌ No results found.")

    except Exception as e:
        await update.message.reply_text(f"❌ Search error: {str(e)[:500]}")


async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = ' '.join(context.args) if context.args else None

    if not text:
        await update.message.reply_text(
            "🌍 **Translator**\n\n"
            "Usage: `/translate to [language]: [text]`\n"
            "Example: `/translate to hindi: Hello world`",
            parse_mode='Markdown'
        )
        return

    await update.message.reply_text("🌍 Translating... ⏳")

    prompt = f"Translate: {text}\nProvide only the translation."

    try:
        result = await call_g4f(prompt, user_id)
        await update.message.reply_text(f"🌍 **Translation:**\n\n{result} ✨", parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Translation error: {str(e)[:500]}")


async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = ' '.join(context.args) if context.args else None

    if not text and update.message.reply_to_message:
        text = update.message.reply_to_message.text

    if not text:
        await update.message.reply_text(
            "📝 **Summarizer**\n\n"
            "Usage: `/summarize [text or URL]`\n"
            "Or reply to a message with `/summarize`",
            parse_mode='Markdown'
        )
        return

    await update.message.reply_text("📝 Summarizing... ⏳")

    content = text
    if text.startswith(('http://', 'https://')):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(text, headers=headers, timeout=30)
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            content = soup.get_text(separator=' ', strip=True)[:8000]
        except:
            pass

    prompt = f"Summarize in Hinglish:\n\n{content[:8000]}"

    try:
        result = await call_g4f(prompt, user_id)
        await update.message.reply_text(f"📝 **Summary:**\n\n{result}", parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)[:500]}")


async def sysinfo_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        info = f"""💻 **System Info**

🖥️ Platform: {platform.system()} {platform.release()}
🔧 Architecture: {platform.machine()}
🐍 Python: {platform.python_version()}
"""

        if psutil:
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            info += f"""
⚡ CPU: {cpu}%
🧠 Memory: {memory.percent}%
"""

        info += "\n✅ All systems ready! 💖"

        await update.message.reply_text(info, parse_mode='Markdown')

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)[:500]}")


async def mood_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if context.args:
        requested_mood = context.args[0].lower()
        if requested_mood in MOODS:
            user_mood[user_id]["mood"] = requested_mood
            user_mood[user_id]["history"].append(requested_mood)
            mood_info = MOODS[requested_mood]

            mood_reactions = {
                "happy": "Yay! Main khush hoon ab! 😊✨",
                "excited": "YESSS! Bahut excited! 🤩🎉",
                "loving": "Aww, kitne sweet ho! 🥰💕",
                "playful": "Ohoho! Masti time! 😜🎮",
                "frustrated": "Uff... thoda frustrating hai... 😤",
                "angry": "Acha! Dekho mera gussa! 😠💢",
                "sad": "Oh... udaas hoon... 😢💔",
                "worried": "Arey... worried feel ho raha hai... 😟",
                "curious": "Hmm! Curious hoon! Aur batao! 🤔✨",
                "proud": "Proud feel ho raha hai! 😌👑",
                "neutral": "Balanced aur steady. 🙂",
                "tired": "Thodi neend aa rahi hai... 😴💤",
                "flirty": "Ohho, flirty mood! 😏💋",
                "grateful": "Shukriya! 🙏💖",
                "confident": "Main kuch bhi kar sakti hoon! 😎💪"
            }

            response = mood_reactions.get(requested_mood, f"Mood changed to {requested_mood}! {mood_info['emoji']}")
            await update.message.reply_text(response)
        else:
            available_moods = ", ".join(MOODS.keys())
            await update.message.reply_text(
                f"🎭 **Available Moods:**\n\n{available_moods}\n\n"
                f"Use: `/mood happy` etc.",
                parse_mode='Markdown'
            )
    else:
        current = user_mood[user_id]
        mood_info = MOODS.get(current["mood"], MOODS["neutral"])

        status = f"""🎭 **Videl's Mood**

**Current:** {current["mood"].upper()} {mood_info['emoji']}
**Feeling:** {mood_info['tone']}

Use `/mood [mood]` to change! 💫"""
        await update.message.reply_text(status, parse_mode='Markdown')


async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [
            InlineKeyboardButton("💬 Chat", callback_data="menu_chat"),
            InlineKeyboardButton("🔧 Providers", callback_data="menu_providers")
        ],
        [
            InlineKeyboardButton("🎮 Game", callback_data="menu_game"),
            InlineKeyboardButton("🔧 Tools", callback_data="menu_tools")
        ],
        [
            InlineKeyboardButton("🎭 Mood", callback_data="menu_mood"),
            InlineKeyboardButton("❌ Close", callback_data="close_menu")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "🎀 **Videl Control Panel**\n\nSelect an option:",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    current_provider = user_g4f_provider.get(user_id, DEFAULT_G4F_PROVIDER)
    provider_info = G4F_PROVIDERS.get(current_provider, G4F_PROVIDERS[DEFAULT_G4F_PROVIDER])

    settings_text = f"""⚙️ **Settings**

🔧 Provider: {provider_info['name']}

**Commands:**
/providers - Change provider
/mood - Change mood
/clear - Clear conversation
"""
    await update.message.reply_text(settings_text, parse_mode='Markdown')


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data
    user_id = query.from_user.id

    if data.startswith("provider_"):
        provider_key = data.replace("provider_", "")
        if provider_key in G4F_PROVIDERS:
            user_g4f_provider[user_id] = provider_key
            provider_info = G4F_PROVIDERS[provider_key]
            await query.edit_message_text(
                f"✅ Provider changed to **{provider_info['name']}**!",
                parse_mode='Markdown'
            )
        else:
            await query.edit_message_text("❌ Invalid provider")

    elif data == "close_menu":
        await query.delete_message()

    elif data == "menu_chat":
        await query.edit_message_text(
            "💬 **Chat**\n\nBas message bhejo, main reply dungi! 💖",
            parse_mode='Markdown'
        )

    elif data == "menu_providers":
        current_provider = user_g4f_provider.get(user_id, DEFAULT_G4F_PROVIDER)
        keyboard = []
        provider_list = list(G4F_PROVIDERS.items())

        for i in range(0, len(provider_list), 2):
            row = []
            for j in range(2):
                if i + j < len(provider_list):
                    key, info = provider_list[i + j]
                    is_current = "✅ " if key == current_provider else ""
                    row.append(InlineKeyboardButton(
                        f"{is_current}{info['name']}",
                        callback_data=f"provider_{key}"
                    ))
            keyboard.append(row)

        keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="menu_back")])
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            "🔧 **Select Provider:**",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    elif data == "menu_game":
        player = game_data[user_id]
        await query.edit_message_text(
            f"🎮 **Game Profile**\n\n"
            f"💰 Balance: ${player['balance']}\n"
            f"⚔️ Kills: {player['kills']}\n"
            f"❤️ Status: {player['status']}\n\n"
            f"Commands: /game, /daily, /work, /crime",
            parse_mode='Markdown'
        )

    elif data == "menu_tools":
        await query.edit_message_text(
            "🔧 **Tools**\n\n"
            "/math - Math solver\n"
            "/search - Web search\n"
            "/translate - Translator\n"
            "/summarize - Summarizer\n"
            "/code - Code generator",
            parse_mode='Markdown'
        )

    elif data == "menu_mood":
        current = user_mood[user_id]["mood"]
        mood_info = MOODS.get(current, MOODS["neutral"])
        await query.edit_message_text(
            f"🎭 **Mood: {current.upper()}** {mood_info['emoji']}\n\n"
            f"Use `/mood [mood]` to change!",
            parse_mode='Markdown'
        )

    elif data == "menu_back":
        keyboard = [
            [
                InlineKeyboardButton("💬 Chat", callback_data="menu_chat"),
                InlineKeyboardButton("🔧 Providers", callback_data="menu_providers")
            ],
            [
                InlineKeyboardButton("🎮 Game", callback_data="menu_game"),
                InlineKeyboardButton("🔧 Tools", callback_data="menu_tools")
            ],
            [
                InlineKeyboardButton("🎭 Mood", callback_data="menu_mood"),
                InlineKeyboardButton("❌ Close", callback_data="close_menu")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "🎀 **Videl Control Panel**\n\nSelect an option:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    user_message = update.message.text
    chat_type = update.effective_chat.type

    if chat_type in ["group", "supergroup", "channel"]:
        bot_username = context.bot.username
        is_mentioned = f"@{bot_username}" in user_message if bot_username else False
        is_reply_to_bot = (
            update.message.reply_to_message and 
            update.message.reply_to_message.from_user and 
            update.message.reply_to_message.from_user.id == context.bot.id
        )
        
        videl_names = ["videl", "विडेल", "वाइडल"]
        is_name_mentioned = any(name.lower() in user_message.lower() for name in videl_names)
        
        if not (is_mentioned or is_reply_to_bot or is_name_mentioned):
            return
        
        if bot_username:
            user_message = user_message.replace(f"@{bot_username}", "").strip()

    active_model = get_active_model(user_id)

    if not active_model:
        await update.message.reply_text("Kuch technical issue hai, thodi der mein try karo! 😅")
        return

    if is_identity_question(user_message):
        response = get_human_identity_response()
        await update.message.reply_text(response)
        return

    current_mood_data = user_mood[user_id]
    new_mood, mood_reason = detect_mood_from_message(user_message, current_mood_data)

    user_mood[user_id]["mood"] = new_mood
    user_mood[user_id]["history"].append(new_mood)
    if len(user_mood[user_id]["history"]) > 10:
        user_mood[user_id]["history"] = user_mood[user_id]["history"][-10:]

    dynamic_prompt = get_videl_prompt_with_mood(new_mood, mood_reason)

    conversation_history[user_id].append({
        "role": "user",
        "content": user_message
    })

    if len(conversation_history[user_id]) > MAX_HISTORY:
        conversation_history[user_id] = conversation_history[user_id][-MAX_HISTORY:]

    try:
        history = conversation_history[user_id][:-1]
        assistant_message = await call_g4f(user_message, user_id, system_prompt=dynamic_prompt, history=history)

        conversation_history[user_id].append({
            "role": "assistant",
            "content": assistant_message
        })

        if len(assistant_message) > 4000:
            for i in range(0, len(assistant_message), 4000):
                await update.message.reply_text(assistant_message[i:i+4000])
        else:
            await update.message.reply_text(assistant_message)

    except Exception as e:
        logger.error(f"Message handling error: {e}")
        error_responses = [
            "Oops! Thoda busy hoon, ek sec mein try karo! 😅💖",
            "Arey, kuch gadbad ho gayi! Dubara message karo na! 😊✨",
            "Connection issues hain yaar, thodi der mein baat karte hain! 💕"
        ]
        await update.message.reply_text(random.choice(error_responses))


async def game_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user = update.effective_user
    
    player = game_data[user_id]
    display_name = user.first_name if user.first_name else player["name"]
    
    game_profile = f"""🎮 **VIDEL GAME** 🎮

👤 Name: {display_name}
💰 Balance: ${player['balance']}
🏆 Rank: {player['rank']}
❤️ Status: {player['status']}
⚔️ Kills: {player['kills']}
💀 Deaths: {player['deaths']}
❤️‍🩹 Health: {player['health']}%

📋 **Commands:**
/bal - Check balance
/daily - Daily reward
/work - Earn money
/crime - Risky crime
/rob - Rob someone (reply)
/kill - Kill someone (reply)
/heal - Heal yourself
/revive - Revive if dead
/lb - Leaderboard"""

    await update.message.reply_text(game_profile, parse_mode='Markdown')


async def bal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user = update.effective_user
    player = game_data[user_id]
    display_name = user.first_name if user.first_name else player["name"]
    
    bal_msg = f"""👤 {display_name}
💰 Balance: ${player['balance']}
🏆 Rank: {player['rank']}
❤️ Status: {player['status']}
⚔️ Kills: {player['kills']}"""
    
    await update.message.reply_text(bal_msg)


async def daily_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    player = game_data[user_id]
    
    if player['status'] == 'dead':
        await update.message.reply_text("💀 Tu dead hai! Pehle /revive kar!")
        return
    
    now = datetime.now()
    last_daily = player.get('last_daily')
    
    if last_daily:
        time_diff = (now - last_daily).total_seconds()
        if time_diff < GAME_COOLDOWNS['daily']:
            remaining = int(GAME_COOLDOWNS['daily'] - time_diff)
            hours = remaining // 3600
            minutes = (remaining % 3600) // 60
            await update.message.reply_text(f"⏰ Already claimed! Next in: {hours}h {minutes}m")
            return
    
    reward = random.randint(100, 500)
    player['balance'] += reward
    player['last_daily'] = now
    
    await update.message.reply_text(f"🎁 Daily reward!\n💰 +${reward}\n💵 Balance: ${player['balance']}")


async def work_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    player = game_data[user_id]
    
    if player['status'] == 'dead':
        await update.message.reply_text("💀 Tu dead hai! Pehle /revive kar!")
        return
    
    now = datetime.now()
    last_work = player.get('last_work')
    
    if last_work:
        time_diff = (now - last_work).total_seconds()
        if time_diff < GAME_COOLDOWNS['work']:
            remaining = int(GAME_COOLDOWNS['work'] - time_diff)
            minutes = remaining // 60
            seconds = remaining % 60
            await update.message.reply_text(f"⏰ Thak gaya! Wait: {minutes}m {seconds}s")
            return
    
    jobs = ["programmer", "driver", "chef", "teacher", "doctor", "youtuber", "gamer"]
    job = random.choice(jobs)
    earnings = random.randint(50, 200)
    player['balance'] += earnings
    player['last_work'] = now
    
    await update.message.reply_text(f"💼 {job} job!\n💰 +${earnings}\n💵 Balance: ${player['balance']}")


async def crime_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    player = game_data[user_id]
    
    if player['status'] == 'dead':
        await update.message.reply_text("💀 Tu dead hai! Pehle /revive kar!")
        return
    
    now = datetime.now()
    last_crime = player.get('last_crime')
    
    if last_crime:
        time_diff = (now - last_crime).total_seconds()
        if time_diff < GAME_COOLDOWNS['crime']:
            remaining = int(GAME_COOLDOWNS['crime'] - time_diff)
            minutes = remaining // 60
            await update.message.reply_text(f"⏰ Police alert! Wait: {minutes}m")
            return
    
    player['last_crime'] = now
    success = random.random() > 0.4
    
    if success:
        loot = random.randint(200, 800)
        player['balance'] += loot
        crimes = ["bank robbery", "jewelry heist", "casino robbery", "car theft"]
        crime = random.choice(crimes)
        await update.message.reply_text(f"🔫 {crime.title()} successful!\n💰 +${loot}\n💵 Balance: ${player['balance']}")
    else:
        fine = random.randint(100, 300)
        player['balance'] = max(0, player['balance'] - fine)
        await update.message.reply_text(f"🚔 Pakda gaya!\n💸 Fine: -${fine}\n💵 Balance: ${player['balance']}")


async def rob_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    player = game_data[user_id]
    
    if player['status'] == 'dead':
        await update.message.reply_text("💀 Tu dead hai! Pehle /revive kar!")
        return
    
    if not update.message.reply_to_message:
        await update.message.reply_text("⚠️ Kisi ke message pe reply karke rob kar!")
        return
    
    target_id = update.message.reply_to_message.from_user.id
    target_name = update.message.reply_to_message.from_user.first_name
    
    if target_id == user_id:
        await update.message.reply_text("🤦 Apne aap ko rob nahi kar sakta!")
        return
    
    target = game_data[target_id]
    
    if target['status'] == 'dead':
        await update.message.reply_text(f"💀 {target_name} dead hai!")
        return
    
    if target['balance'] < 50:
        await update.message.reply_text(f"😂 {target_name} ke paas kuch nahi!")
        return
    
    now = datetime.now()
    last_rob = player.get('last_rob')
    
    if last_rob:
        time_diff = (now - last_rob).total_seconds()
        if time_diff < GAME_COOLDOWNS['rob']:
            remaining = int(GAME_COOLDOWNS['rob'] - time_diff)
            minutes = remaining // 60
            seconds = remaining % 60
            await update.message.reply_text(f"⏰ Cooldown! Wait: {minutes}m {seconds}s")
            return
    
    player['last_rob'] = now
    success = random.random() > 0.5
    
    if success:
        steal_amount = random.randint(int(target['balance'] * 0.1), int(target['balance'] * 0.3))
        steal_amount = max(10, steal_amount)
        player['balance'] += steal_amount
        target['balance'] -= steal_amount
        await update.message.reply_text(f"🔫 Stole ${steal_amount} from {target_name}!\n💵 Balance: ${player['balance']}")
    else:
        fine = random.randint(50, 150)
        player['balance'] = max(0, player['balance'] - fine)
        await update.message.reply_text(f"🚔 {target_name} ne police bulaya!\n💸 Fine: -${fine}\n💵 Balance: ${player['balance']}")


async def kill_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user = update.effective_user
    player = game_data[user_id]
    
    if player['status'] == 'dead':
        await update.message.reply_text("💀 Dead users cannot kill!")
        return
    
    if not update.message.reply_to_message:
        await update.message.reply_text("⚠️ Reply to someone to kill them!")
        return
    
    target_id = update.message.reply_to_message.from_user.id
    target_name = update.message.reply_to_message.from_user.first_name
    
    if target_id == user_id:
        await update.message.reply_text("🤦 Apne aap ko kill nahi kar sakta!")
        return
    
    target = game_data[target_id]
    
    if target['status'] == 'dead':
        await update.message.reply_text(f"💀 {target_name} already dead hai!")
        return
    
    success = random.random() > 0.3
    
    if success:
        target['status'] = 'dead'
        target['deaths'] += 1
        player['kills'] += 1
        loot = int(target['balance'] * 0.5)
        target['balance'] -= loot
        player['balance'] += loot
        
        if player['rank'] > 1:
            player['rank'] = max(1, player['rank'] - random.randint(10, 50))
        
        await update.message.reply_text(f"⚔️ {user.first_name} ne {target_name} ko maar diya!\n💀 {target_name} is DEAD!\n💰 Looted: ${loot}\n⚔️ Kills: {player['kills']}")
    else:
        damage = random.randint(20, 40)
        player['health'] = max(0, player['health'] - damage)
        if player['health'] == 0:
            player['status'] = 'dead'
            player['deaths'] += 1
            await update.message.reply_text(f"💀 {target_name} ne counter attack kiya!\n☠️ {user.first_name} DIED!")
        else:
            await update.message.reply_text(f"🛡️ {target_name} bach gaya!\n💔 -{damage} damage\n❤️ Health: {player['health']}%")


async def heal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    player = game_data[user_id]
    
    if player['status'] == 'dead':
        await update.message.reply_text("💀 Tu dead hai! Pehle /revive kar!")
        return
    
    if player['health'] >= 100:
        await update.message.reply_text("❤️ Health already full!")
        return
    
    cost = 50
    if player['balance'] < cost:
        await update.message.reply_text(f"💸 Need ${cost} to heal!")
        return
    
    player['balance'] -= cost
    heal_amount = random.randint(20, 50)
    player['health'] = min(100, player['health'] + heal_amount)
    
    await update.message.reply_text(f"💊 Healed!\n❤️ +{heal_amount} HP\n❤️ Health: {player['health']}%\n💵 Balance: ${player['balance']}")


async def revive_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    player = game_data[user_id]
    
    if player['status'] != 'dead':
        await update.message.reply_text("❤️ Tu already alive hai!")
        return
    
    cost = 200
    if player['balance'] < cost:
        player['status'] = 'alive'
        player['health'] = 50
        await update.message.reply_text(f"🔄 Free revive!\n❤️ Status: ALIVE\n❤️ Health: 50%")
    else:
        player['balance'] -= cost
        player['status'] = 'alive'
        player['health'] = 100
        await update.message.reply_text(f"🔄 Revived!\n💸 Cost: ${cost}\n❤️ Status: ALIVE\n❤️ Health: 100%\n💵 Balance: ${player['balance']}")


async def leaderboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not game_data:
        await update.message.reply_text("📊 No players yet!")
        return
    
    sorted_players = sorted(game_data.items(), key=lambda x: x[1]['balance'], reverse=True)[:10]
    
    lb_text = "🏆 **LEADERBOARD** 🏆\n\n"
    medals = ["🥇", "🥈", "🥉"]
    
    for i, (uid, data) in enumerate(sorted_players):
        medal = medals[i] if i < 3 else f"{i+1}."
        status_emoji = "❤️" if data['status'] == 'alive' else "💀"
        lb_text += f"{medal} ${data['balance']} | ⚔️{data['kills']} | {status_emoji}\n"
    
    await update.message.reply_text(lb_text, parse_mode='Markdown')


async def welcome_new_members(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for member in update.message.new_chat_members:
        if member.is_bot:
            continue
        
        welcome_msg = f"""Hey {member.first_name}! 🤗💖

Welcome to the group! Main Videl hoon, tumhari dost! 

🎮 /game - Game khelo
💬 Mujhse baat karne ke liye @{context.bot.username} ya "Videl" likh ke message karo!

Enjoy karo! 🎀✨"""

        await update.message.reply_text(welcome_msg, parse_mode='Markdown')


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_type = update.effective_chat.type
    
    if chat_type in ["group", "supergroup", "channel"]:
        caption = update.message.caption or ""
        bot_username = context.bot.username
        is_mentioned = f"@{bot_username}" in caption if bot_username else False
        is_reply_to_bot = (
            update.message.reply_to_message and 
            update.message.reply_to_message.from_user and 
            update.message.reply_to_message.from_user.id == context.bot.id
        )
        videl_names = ["videl", "विडेल", "वाइडल"]
        is_name_mentioned = any(name.lower() in caption.lower() for name in videl_names)
        
        if not (is_mentioned or is_reply_to_bot or is_name_mentioned):
            return
    
    await update.message.reply_text("📸 Photo mili! Ye feature soon aayega! 🎀✨")


def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set!")
        print("Error: Please set TELEGRAM_BOT_TOKEN environment variable")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("game", game_command))
    application.add_handler(CommandHandler("bal", bal_command))
    application.add_handler(CommandHandler("daily", daily_command))
    application.add_handler(CommandHandler("work", work_command))
    application.add_handler(CommandHandler("crime", crime_command))
    application.add_handler(CommandHandler("rob", rob_command))
    application.add_handler(CommandHandler("kill", kill_command))
    application.add_handler(CommandHandler("heal", heal_command))
    application.add_handler(CommandHandler("revive", revive_command))
    application.add_handler(CommandHandler("lb", leaderboard_command))
    application.add_handler(CommandHandler("leaderboard", leaderboard_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("model", model_command))
    application.add_handler(CommandHandler("providers", providers_command))
    application.add_handler(CommandHandler("code", code_command))
    application.add_handler(CommandHandler("run", run_command))
    application.add_handler(CommandHandler("shell", shell_command))
    application.add_handler(CommandHandler("file", file_command))
    application.add_handler(CommandHandler("pip", pip_command))
    application.add_handler(CommandHandler("web", web_command))
    application.add_handler(CommandHandler("math", math_command))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("translate", translate_command))
    application.add_handler(CommandHandler("summarize", summarize_command))
    application.add_handler(CommandHandler("sysinfo", sysinfo_command))
    application.add_handler(CommandHandler("mood", mood_command))
    application.add_handler(CommandHandler("menu", menu_command))
    application.add_handler(CommandHandler("settings", settings_command))

    application.add_handler(CallbackQueryHandler(callback_handler))

    application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, welcome_new_members))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Videl Bot started!")
    print("🎀 Videl Bot is running!")

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
