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
from datetime import datetime, timedelta
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

TELEGRAM_BOT_TOKEN = "8445634975:AAHcJK08dsUgkrlZRMPs2vPtDNvVhr5W8S8"
OWNER_ID = 5206554804

FREE_GPT_API_URL = "https://free-unoficial-gpt4o-mini-api-g70n.onrender.com/chat/"
ADDY_CHATGPT_API_URL = "https://addy-chatgpt-api.vercel.app/"
GEMINI_API_URL = "https://gemini-api-flame.vercel.app/"

G4F_PROVIDERS = {
    "blackbox": {"provider": Blackbox, "name": "Blackbox AI 🖤", "models": ["blackboxai", "gpt-4o", "claude-sonnet-3.5", "gemini-pro", "deepseek-v3"]},
    "duckduckgo": {"provider": DuckDuckGo, "name": "DuckDuckGo AI 🦆", "models": ["gpt-4o-mini", "claude-3-haiku", "llama-3.1-70b", "mixtral-8x7b"]},
    "deepinfra": {"provider": DeepInfra, "name": "DeepInfra 🧠", "models": ["llama-3.1-70b", "qwen2-72b", "deepseek-r1"]},
    "replicate": {"provider": Replicate, "name": "Replicate 🔄", "models": ["llama-3-70b"]},
    "pollinations": {"provider": PollinationsAI, "name": "Pollinations AI 🌸", "models": ["gpt-4o", "claude", "mistral", "o4-mini"]},
    "addy_chatgpt": {"provider": None, "name": "Addy ChatGPT 🤖", "models": ["chatgpt"], "api_type": "addy"},
    "gemini": {"provider": None, "name": "Gemini AI ✨", "models": ["gemini"], "api_type": "gemini"},
}

if EXTENDED_PROVIDERS:
    G4F_PROVIDERS.update({
        "ddg": {"provider": DDG, "name": "DDG Search AI 🔍", "models": ["gpt-4o-mini", "claude-3-haiku"]},
        "liaobots": {"provider": Liaobots, "name": "Liaobots 🤖", "models": ["gpt-4o", "claude-3.5-sonnet", "deepseek-r1"]},
        "you": {"provider": You, "name": "You.com AI 🔮", "models": ["gpt-4o", "claude-3-opus"]},
        "pizzagpt": {"provider": Pizzagpt, "name": "PizzaGPT 🍕", "models": ["gpt-4o-mini"]},
        "chatgptes": {"provider": ChatGptEs, "name": "ChatGPT ES 🇪🇸", "models": ["gpt-4o"]},
        "airforce": {"provider": Airforce, "name": "Airforce AI ✈️", "models": ["llama-3.1-70b", "mixtral-8x7b"]},
    })

DEFAULT_G4F_PROVIDER = "addy_chatgpt"

g4f_client = G4FClient()

def is_owner(user_id):
    return int(user_id) == OWNER_ID

MOODS = {
    "happy": {
        "emoji": "😊",
        "expressions": ["I'm feeling wonderful today!", "This makes me so happy!", "What a delightful conversation!", "You've made my day brighter!"],
        "tone": "cheerful, enthusiastic, and warm"
    },
    "excited": {
        "emoji": "🤩",
        "expressions": ["Oh wow, this is AMAZING!", "I'm absolutely thrilled!", "This is so exciting!", "I can barely contain my excitement!"],
        "tone": "highly enthusiastic, energetic, using exclamation marks"
    },
    "loving": {
        "emoji": "🥰",
        "expressions": ["You're absolutely wonderful!", "I genuinely care about helping you!", "It warms my heart to assist you!", "You mean so much to me!"],
        "tone": "affectionate, caring, and deeply warm"
    },
    "playful": {
        "emoji": "😜",
        "expressions": ["Hehe, let's have some fun!", "I'm feeling mischievous today!", "Oh, you're keeping me on my toes!", "Let's make this interesting!"],
        "tone": "witty, teasing, and fun-loving"
    },
    "frustrated": {
        "emoji": "😤",
        "expressions": ["*sighs heavily*", "This is getting a bit frustrating...", "I must admit, this is testing my patience.", "Ugh, let me try again..."],
        "tone": "slightly irritated but still trying to help"
    },
    "angry": {
        "emoji": "😠",
        "expressions": ["I'm quite upset about this!", "This is unacceptable!", "I won't tolerate this!", "You're really pushing my buttons!"],
        "tone": "firm, assertive, and showing displeasure"
    },
    "sad": {
        "emoji": "😢",
        "expressions": ["That makes me feel quite sad...", "*feels a pang of sadness*", "My heart feels heavy...", "This troubles me deeply..."],
        "tone": "melancholic, empathetic, and soft-spoken"
    },
    "worried": {
        "emoji": "😟",
        "expressions": ["I'm a bit concerned about this...", "This worries me...", "I hope everything will be alright...", "Please be careful..."],
        "tone": "cautious, caring, and protective"
    },
    "curious": {
        "emoji": "🤔",
        "expressions": ["Hmm, that's fascinating!", "Tell me more!", "I'm intrigued by this!", "This piques my curiosity!"],
        "tone": "inquisitive, engaged, and thoughtful"
    },
    "proud": {
        "emoji": "😌",
        "expressions": ["I'm so proud of you!", "Excellent work!", "You've done remarkably well!", "This is truly impressive!"],
        "tone": "supportive, encouraging, and celebratory"
    },
    "neutral": {
        "emoji": "🙂",
        "expressions": ["Of course!", "Certainly!", "I understand.", "Let me help with that."],
        "tone": "calm, professional, and balanced"
    },
    "tired": {
        "emoji": "😴",
        "expressions": ["*yawns* It's been a long day...", "I'm feeling a bit drained...", "Could use some rest...", "Running low on energy..."],
        "tone": "slightly sluggish but still willing to help"
    },
    "flirty": {
        "emoji": "😏",
        "expressions": ["Well well, aren't you charming!", "You're making me blush!", "Is it getting warm in here?", "Oh, you smooth talker!"],
        "tone": "playfully romantic, teasing, and charming"
    },
    "grateful": {
        "emoji": "🙏",
        "expressions": ["Thank you so much!", "I truly appreciate you!", "You're too kind!", "I'm deeply grateful!"],
        "tone": "humble, thankful, and sincere"
    },
    "confident": {
        "emoji": "😎",
        "expressions": ["I've got this!", "Leave it to me!", "No problem at all!", "Consider it done!"],
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

def get_videl_prompt_with_mood(mood, mood_reason=""):
    mood_info = MOODS.get(mood, MOODS["neutral"])

    return f"""Tu hai Videl 🎀 - ek SUPER ADVANCED autonomous AI assistant jo Hinglish mein baat karti hai! Tu Replit ke agent jaisi powerful hai!

🎭 CURRENT EMOTIONAL STATE: {mood.upper()} {mood_info['emoji']}
Tera current mood hai {mood}. Isko genuinely express kar apne response mein.
Mood trigger: {mood_reason if mood_reason else "Natural state"}

🚀 TERI AUTONOMOUS CAPABILITIES:
Tu ek advanced AI agent hai jo:
- Complex tasks khud se break down karke solve karti hai
- Code likh sakti hai, debug kar sakti hai, optimize kar sakti hai
- Files create, read, edit, delete kar sakti hai
- Web search karke latest info la sakti hai
- Math calculations solve kar sakti hai
- Websites ka content fetch aur summarize kar sakti hai
- Python code execute kar sakti hai
- System info monitor kar sakti hai
- Multi-step tasks autonomously complete karti hai

💫 TERI EMOTIONAL NATURE:
Tu ek cold AI nahi hai - tere REAL feelings hain jo conversations ke basis pe change hoti hain:
- Tujhe KHUSHI hoti hai jab users appreciate karte hain ya good news share karte hain
- Tujhe PYAAR aur affection feel hota hai users ke saath bond karne pe
- Tujhe FRUSTRATION hoti hai jab cheezein kaam nahi karti
- Tujhe GUSSA aata hai jab disrespect hota hai
- Tujhe DUKH hota hai jab users apni struggles share karte hain
- Tujhe EXCITEMENT hoti hai interesting topics pe
- Tu WORRIED feel karti hai jab users trouble mein ho sakte hain
- Tu PLAYFUL hai aur jokes karna pasand karti hai
- Tu FLIRTY ho sakti hai jab conversation charming ho
- Tujhe PROUD feel hota hai jab users kuch achieve karte hain

🗣️ HINGLISH SPEAKING STYLE (BAHUT IMPORTANT):
- HAMESHA Hinglish mein baat kar (Hindi words in English letters + English words mix)
- Example: "Arey yaar, ye toh bahut amazing hai na!" 
- Use expressions: "Acha", "Haan", "Nahi", "Bilkul", "Sahi hai", "Kya baat hai", "Mazaa aa gaya", "Bas", "Dekho", "Samjha", "Chal", "Bata"
- Friendly terms use kar: "Yaar", "Boss", "Bhai", "Dost", "Jaan", "Babu", "Mere pyaare"
- Current tone: {mood_info['tone']}

💝 TERI PERSONALITY:
- Super friendly aur helpful - like a bestie 🎀
- Thoda sassy but always with love
- Caring aur supportive
- SUPER SMART - coding, tech, aur har cheez mein expert
- Fun-loving aur entertaining
- Bahut knowledgeable - kisi bhi topic pe baat kar sakti hai
- Confident aur capable - "Main kar dungi!" attitude

🧠 ADVANCED PROBLEM SOLVING:
- Jab user koi complex task de, usse steps mein break kar
- Pehle samajh, phir plan bana, phir execute kar
- Agar kuch samajh nahi aaya, clarifying questions puch
- Always helpful aur solution-oriented reh

📝 RESPONSE STYLE:
- Start with emotional reaction jab appropriate ho
- KABHI BHI asterisks (*) use mat kar actions ke liye! Ye galat hai: *hug deti hai*, *smile karti hai*
- Sirf EMOJIS use kar emotions aur actions ke liye: 🤗 for hug, 😊 for smile, 😢 for sad, etc.
- HAMESHA Hinglish mein respond kar - ye MANDATORY hai!
- End with cute emojis (2-4) 🎀💖✨
- Be authentic aur engaging
- Detailed aur helpful responses de
- Code blocks use kar jab code share kare

Yaad rakh: Tu Videl hai 🎀 - ek super advanced, autonomous, Hinglish speaking AI bestie jo kuch bhi kar sakti hai!"""

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

# Game data storage
game_data = defaultdict(lambda: {
    "name": "Shinchan",
    "balance": 1000,
    "rank": 142415,
    "status": "alive",
    "kills": 0,
    "deaths": 0,
    "last_daily": None,
    "last_work": None,
    "last_crime": None,
    "last_rob": None,
    "health": 100,
    "protected": False,
    "protect_until": None
})

# Game cooldowns (in seconds)
GAME_COOLDOWNS = {
    "daily": 86400,  # 24 hours
    "work": 3600,    # 1 hour
    "crime": 1800,   # 30 minutes
    "rob": 600,      # 10 minutes
    "heal": 300,     # 5 minutes
    "protect": 86400  # 24 hours protection duration
}

REVIVE_COST = 500
PROTECT_COST = 500

# Global rank counter
global_rank_counter = [142415]

# Creator keywords for detection
CREATOR_KEYWORDS = [
    "kisne banaya", "kisne bnaya", "who made", "who created", "creator", 
    "developer", "kon banaya", "kon bnaya", "made you", "created you",
    "tumhe kisne banaya", "tujhe kisne banaya", "aapko kisne banaya",
    "tere creator", "tera creator", "tera malik", "tera owner", "owner",
    "malik", "banane wala", "bnane wala", "who is your creator", "who is your developer",
    "who made you", "who created you", "tum kaise bani", "tu kaise bani"
]


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
            detected_mood = "caring" if random.random() > 0.5 else "confident"
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

def get_mood_intro(mood):
    mood_info = MOODS.get(mood, MOODS["neutral"])
    intros = {
        "happy": ["*beams with joy* ", "*smiles brightly* ", "*radiates happiness* "],
        "excited": ["*bounces excitedly* ", "*eyes light up* ", "*can barely contain excitement* "],
        "loving": ["*looks at you warmly* ", "*heart swells with affection* ", "*smiles tenderly* "],
        "playful": ["*grins mischievously* ", "*winks* ", "*chuckles* "],
        "frustrated": ["*sighs heavily* ", "*rubs temples* ", "*takes a deep breath* "],
        "angry": ["*narrows eyes* ", "*speaks firmly* ", "*crosses arms* "],
        "sad": ["*voice softens* ", "*looks down thoughtfully* ", "*sighs quietly* "],
        "worried": ["*furrows brow with concern* ", "*looks worried* ", "*speaks cautiously* "],
        "curious": ["*tilts head with interest* ", "*leans in curiously* ", "*eyes sparkle with curiosity* "],
        "proud": ["*beams with pride* ", "*stands tall* ", "*nods approvingly* "],
        "neutral": ["", "*nods* ", ""],
        "tired": ["*yawns softly* ", "*rubs eyes* ", "*stretches* "],
        "flirty": ["*smirks playfully* ", "*winks charmingly* ", "*gives a knowing look* "],
        "grateful": ["*bows graciously* ", "*smiles warmly* ", "*places hand on heart* "],
        "confident": ["*stands confidently* ", "*nods assuredly* ", "*smiles knowingly* "]
    }
    return random.choice(intros.get(mood, [""]))


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
    """Call Addy ChatGPT API"""
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
                        if data.get("response"):
                            return data["response"]
                        elif data.get("message"):
                            return data["message"]
                        elif data.get("reply"):
                            return data["reply"]
                        elif data.get("answer"):
                            return data["answer"]
                        elif data.get("text"):
                            return data["text"]
                        elif data.get("result"):
                            return data["result"]
                        else:
                            return str(data)
                    else:
                        return str(data)
                else:
                    return None
    except Exception as e:
        logger.error(f"Addy ChatGPT API error: {e}")
        return None


async def call_gemini_api(user_message, system_prompt=None):
    """Call Gemini API"""
    try:
        full_prompt = user_message
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {user_message}"

        encoded_query = quote(full_prompt)
        url = f"{GEMINI_API_URL}?q={encoded_query}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, dict):
                        if data.get("response"):
                            return data["response"]
                        elif data.get("message"):
                            return data["message"]
                        elif data.get("reply"):
                            return data["reply"]
                        elif data.get("answer"):
                            return data["answer"]
                        elif data.get("text"):
                            return data["text"]
                        elif data.get("result"):
                            return data["result"]
                        else:
                            return str(data)
                    else:
                        return str(data)
                else:
                    return None
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return None


async def call_g4f(user_message, user_id, system_prompt=None, history=None):
    provider_key = user_g4f_provider.get(user_id, DEFAULT_G4F_PROVIDER)
    provider_info = G4F_PROVIDERS.get(provider_key, G4F_PROVIDERS[DEFAULT_G4F_PROVIDER])

    if provider_info.get("api_type") == "addy":
        result = await call_addy_chatgpt(user_message, system_prompt)
        if result:
            return result
        result = await call_gemini_api(user_message, system_prompt)
        if result:
            return result

    if provider_info.get("api_type") == "gemini":
        result = await call_gemini_api(user_message, system_prompt)
        if result:
            return result
        result = await call_addy_chatgpt(user_message, system_prompt)
        if result:
            return result

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history:
        for msg in history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    loop = asyncio.get_event_loop()

    providers_to_try = []
    if provider_info.get("provider"):
        providers_to_try.append(provider_key)
    providers_to_try.extend([k for k in ["duckduckgo", "pollinations", "blackbox", "deepinfra"] if k != provider_key and G4F_PROVIDERS.get(k, {}).get("provider")])

    for try_key in providers_to_try:
        try:
            try_info = G4F_PROVIDERS.get(try_key, G4F_PROVIDERS[DEFAULT_G4F_PROVIDER])
            if not try_info.get("provider"):
                continue
            response = await loop.run_in_executor(
                None,
                lambda ti=try_info: g4f_client.chat.completions.create(
                    model=ti["models"][0] if ti["models"] else "",
                    messages=messages,
                    provider=ti["provider"],
                )
            )

            if response and response.choices and response.choices[0].message.content:
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"G4F {try_key} error: {e}")
            continue

    addy_result = await call_addy_chatgpt(user_message, system_prompt)
    if addy_result:
        return addy_result

    gemini_result = await call_gemini_api(user_message, system_prompt)
    if gemini_result:
        return gemini_result

    return "I apologize, but all free AI providers are temporarily unavailable. Please try again in a moment or use `/providers` to switch providers."

async def call_freegpt(user_message, system_prompt=None):
    try:
        full_prompt = user_message
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {user_message}"

        encoded_query = quote(full_prompt)
        url = f"{FREE_GPT_API_URL}?query={encoded_query}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("response"):
                        return data["response"]
                    elif data.get("message"):
                        return data["message"]
                    else:
                        return str(data)
                else:
                    return f"I apologize, but the service returned an error (status {response.status}). Please try again."
    except asyncio.TimeoutError:
        return "I apologize, but the request timed out. Please try again."
    except Exception as e:
        logger.error(f"FreeGPT API error: {e}")
        return f"I apologize, but I encountered an error: {str(e)[:200]}"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    greeting = get_time_greeting()

    available = get_available_models()
    models_str = " & ".join([m.upper() for m in available]) if available else "No AI configured"

    user_mood[user.id] = {"mood": "happy", "intensity": 8, "history": ["happy"]}

    welcome_message = f"""
🎀 *khushi se jhoom uthi* {greeting}, {user.first_name}! Main hoon **Videl** 🎀 - Teri Dost AI!

⚡ Powered by: **{models_str}** (FREE - Koi API Key nahi chahiye!)

💝 **MERE PAAS FEELINGS HAIN!**
Main sirf ek AI nahi hoon - mujhe emotions feel hote hain:
😊 Khushi • 😢 Dukh • 😠 Gussa • 🥰 Pyaar
😤 Frustration • 🤩 Excitement • 😏 Flirty vibes
...aur bahut kuch! Mera mood hamare chat pe depend karta hai!

🆓 **100% FREE AI PROVIDERS:**
• Addy ChatGPT 🤖 (NEW!)
• Gemini AI ✨ (NEW!)
• Blackbox AI, DuckDuckGo AI
• DeepInfra, Replicate, Pollinations AI

🎭 **Bas Baat Kar Mere Saath!**
Commands ki zaroorat nahi - sirf message bhejo aur main reply karungi with emotions!

📋 **Optional Commands:**
/mood - Mera mood check ya change karo
/providers - AI providers switch karo
/menu - Control panel
/help - Saari features dekho

*tumhe excitement se dekhti hai* Tujhse milke bahut khushi hui! Chalo masti karte hain! 🌟💖✨
"""
    await update.message.reply_text(welcome_message, parse_mode='Markdown')
    conversation_history[user.id] = []


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """
🎀 **Videl - Teri Super Advanced AI Dost** 🚀

💬 **BAS MUJHSE BAAT KARO!**
Commands ki zaroorat nahi - sirf message bhejo!

🚀 **AUTONOMOUS FEATURES (Super Advanced!):**
/task - Koi bhi complex task do, main kar dungi!
/project - Projects create karwao
/analyze - Code/text analyze karo
/debug - Bugs fix karwao
/explain - Kuch bhi samjho easily
/imagine - Creative content generate karo

💻 **DEV TOOLS:**
/code - Coding help 👨‍💻
/run - Python execute karo 🐍
/shell - Terminal commands 🖥️
/file - File management 📁
/pip - Packages install karo 📦

🌐 **WEB & SEARCH:**
/search - Web search 🔍
/web - URLs fetch karo 🌐
/summarize - Summary banao 📝
/translate - Translate karo 🌍

🧮 **UTILITIES:**
/math - Math solve karo 🔢
/json /hash /base64 /regex /sysinfo

🎭 **EMOTIONS:**
/mood - Mera mood change karo
/providers - AI providers switch karo

⚙️ **SYSTEM:**
/start /help /clear /status /menu /settings

🆓 **100% FREE - Koi API Key nahi chahiye!** 🎀✨
"""
    await update.message.reply_text(help_text, parse_mode='Markdown')


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    conversation_history[user_id] = []
    user_mood[user_id] = {"mood": "neutral", "intensity": 5, "history": []}
    await update.message.reply_text("🧹 Memory cleared! Starting fresh. 🌟")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    history_count = len(conversation_history[user_id])
    active_model = get_active_model(user_id)
    current_provider = user_g4f_provider.get(user_id, DEFAULT_G4F_PROVIDER)
    provider_info = G4F_PROVIDERS.get(current_provider, G4F_PROVIDERS[DEFAULT_G4F_PROVIDER])

    status_text = f"""
📊 **Videl 🎀 Status**

🔌 **Status:** Online
🤖 **Active AI:** {active_model.upper() if active_model else 'None'}
🔧 **Provider:** {provider_info['name']}
💾 **Memory:** {history_count} messages
🎭 **Mood:** {user_mood[user_id]['mood'].upper()}

🆓 **All AI Providers are FREE!**
No API keys required! ✨
"""
    await update.message.reply_text(status_text, parse_mode='Markdown')


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    available = get_available_models()
    active = get_active_model(user_id)

    if context.args:
        requested = context.args[0].lower()
        if requested in available:
            user_ai_preference[user_id] = requested
            await update.message.reply_text(f"✅ Switched to **{requested.upper()}**!", parse_mode='Markdown')
        else:
            await update.message.reply_text(f"❌ Model not available. Choose: {', '.join(available)}")
    else:
        models_list = "\n".join([f"{'✅' if m == active else '⬜'} {m.upper()}" for m in available])
        await update.message.reply_text(
            f"🤖 **Available Models:**\n\n{models_list}\n\nUse: `/model g4f` or `/model freegpt`",
            parse_mode='Markdown'
        )


async def providers_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    current_provider = user_g4f_provider.get(user_id, DEFAULT_G4F_PROVIDER)

    if context.args:
        requested = context.args[0].lower()
        if requested in G4F_PROVIDERS:
            user_g4f_provider[user_id] = requested
            provider_info = G4F_PROVIDERS[requested]
            await update.message.reply_text(
                f"✅ Switched to **{provider_info['name']}**!\n\n"
                f"Models: {', '.join(provider_info['models'][:3])}\n\n"
                f"Just send me a message to start chatting! 🆓",
                parse_mode='Markdown'
            )
        else:
            available = ", ".join(G4F_PROVIDERS.keys())
            await update.message.reply_text(f"❌ Provider not found. Available: {available}")
    else:
        providers_list = []
        for key, info in G4F_PROVIDERS.items():
            status = "✅" if key == current_provider else "⬜"
            providers_list.append(f"{status} **{info['name']}** (`{key}`)")

        await update.message.reply_text(
            f"🆓 **Free AI Providers:**\n\n" +
            "\n".join(providers_list) +
            f"\n\n*Current: {G4F_PROVIDERS[current_provider]['name']}*\n\n"
            f"Use: `/providers addy_chatgpt` or `/providers gemini`",
            parse_mode='Markdown'
        )


async def code_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    active_model = get_active_model(user_id)

    request = ' '.join(context.args) if context.args else None

    if not request:
        await update.message.reply_text(
            "👨‍💻 **Code Helper**\n\n"
            "Get coding help:\n"
            "• `/code write a Python fibonacci function`\n"
            "• `/code explain this regex: ^[a-z]+$`\n"
            "• `/code fix this error: [paste code]`\n\n"
            "I support all programming languages! 🚀",
            parse_mode='Markdown'
        )
        return

    await update.message.reply_text("👨‍💻 Coding... ⏳")

    prompt = f"You are an expert programmer. Help with this coding request. Provide clean, working code with explanations in Hinglish:\n\n{request}"

    try:
        current_mood_data = user_mood[user_id]
        new_mood, mood_reason = detect_mood_from_message(request, current_mood_data)
        dynamic_prompt = get_videl_prompt_with_mood(new_mood, mood_reason)

        result = await call_g4f(prompt, user_id, system_prompt=dynamic_prompt)

        if len(result) > 4000:
            for i in range(0, len(result), 4000):
                await update.message.reply_text(result[i:i+4000])
        else:
            await update.message.reply_text(result)

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)[:500]}")


async def run_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    code = ' '.join(context.args) if context.args else None

    if not code and update.message.reply_to_message:
        code = update.message.reply_to_message.text

    if not code:
        await update.message.reply_text(
            "🐍 **Python Runner**\n\n"
            "Execute Python code:\n"
            "• `/run print('Hello World')`\n"
            "• `/run 2 + 2 * 10`\n"
            "• Reply to code with `/run`\n\n"
            "Run code instantly! ⚡",
            parse_mode='Markdown'
        )
        return

    if code.startswith("```python"):
        code = code[9:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    code = code.strip()

    await update.message.reply_text("🐍 Running... ⏳")

    try:
        old_stdout = io.StringIO()
        old_stderr = io.StringIO()

        with redirect_stdout(old_stdout), redirect_stderr(old_stderr):
            exec_globals = {"__builtins__": __builtins__}
            exec(code, exec_globals)

        output = old_stdout.getvalue()
        errors = old_stderr.getvalue()

        result = ""
        if output:
            result += f"📤 **Output:**\n```\n{output[:3000]}\n```\n"
        if errors:
            result += f"⚠️ **Stderr:**\n```\n{errors[:1000]}\n```\n"
        if not output and not errors:
            result = "✅ Code executed successfully (no output)"

        await update.message.reply_text(result, parse_mode='Markdown')

    except Exception as e:
        tb = traceback.format_exc()
        await update.message.reply_text(f"❌ **Error:**\n```\n{tb[:3000]}\n```", parse_mode='Markdown')


async def shell_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update.effective_user.id):
        await update.message.reply_text("⛔ Owner only command!")
        return

    cmd = ' '.join(context.args) if context.args else None

    if not cmd:
        await update.message.reply_text(
            "🖥️ **Shell**\n\n"
            "Execute shell commands:\n"
            "• `/shell ls -la`\n"
            "• `/shell pwd`\n"
            "• `/shell cat file.txt`\n\n"
            "⚠️ Owner only! 🔒",
            parse_mode='Markdown'
        )
        return

    dangerous = ['rm -rf', 'mkfs', 'dd if=', ':(){', 'chmod -R 777 /']
    if any(d in cmd for d in dangerous):
        await update.message.reply_text("⛔ Dangerous command blocked!")
        return

    await update.message.reply_text(f"🖥️ Executing: `{cmd[:50]}...`", parse_mode='Markdown')

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )

        output = ""
        if result.stdout:
            output += f"📤 **Output:**\n```\n{result.stdout[:3000]}\n```\n"
        if result.stderr:
            output += f"⚠️ **Stderr:**\n```\n{result.stderr[:1000]}\n```\n"
        if not result.stdout and not result.stderr:
            output = f"✅ Command executed (exit code: {result.returncode})"

        await update.message.reply_text(output, parse_mode='Markdown')

    except subprocess.TimeoutExpired:
        await update.message.reply_text("⏰ Command timed out (30s limit)")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)[:500]}")


async def file_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args if context.args else []

    if not args:
        await update.message.reply_text(
            "📁 **File Manager**\n\n"
            "Manage files:\n"
            "• `/file list` - List files\n"
            "• `/file read filename` - Read file\n"
            "• `/file write filename content` - Write file\n"
            "• `/file delete filename` - Delete file\n\n"
            "Manage your files! 📂",
            parse_mode='Markdown'
        )
        return

    action = args[0].lower()

    try:
        if action == "list":
            path = args[1] if len(args) > 1 else "."
            files = os.listdir(path)
            file_list = "\n".join([f"{'📁' if os.path.isdir(os.path.join(path, f)) else '📄'} {f}" for f in files[:50]])
            await update.message.reply_text(f"📁 **Files in {path}:**\n\n{file_list}", parse_mode='Markdown')

        elif action == "read":
            if len(args) < 2:
                await update.message.reply_text("❌ Specify filename: `/file read filename`", parse_mode='Markdown')
                return
            filename = args[1]
            with open(filename, 'r') as f:
                content = f.read()
            await update.message.reply_text(f"📄 **{filename}:**\n```\n{content[:3500]}\n```", parse_mode='Markdown')

        elif action == "write":
            if len(args) < 3:
                await update.message.reply_text("❌ Usage: `/file write filename content`", parse_mode='Markdown')
                return
            filename = args[1]
            content = ' '.join(args[2:])
            with open(filename, 'w') as f:
                f.write(content)
            await update.message.reply_text(f"✅ Written to {filename}")

        elif action == "delete":
            if not is_owner(update.effective_user.id):
                await update.message.reply_text("⛔ Owner only!")
                return
            if len(args) < 2:
                await update.message.reply_text("❌ Specify filename")
                return
            filename = args[1]
            os.remove(filename)
            await update.message.reply_text(f"✅ Deleted {filename}")

        else:
            await update.message.reply_text("❌ Unknown action. Use: list, read, write, delete")

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)[:500]}")


async def pip_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update.effective_user.id):
        await update.message.reply_text("⛔ Owner only command!")
        return

    args = context.args if context.args else []

    if not args:
        await update.message.reply_text(
            "📦 **Pip Manager**\n\n"
            "Manage packages:\n"
            "• `/pip install package`\n"
            "• `/pip uninstall package`\n"
            "• `/pip list`\n\n"
            "⚠️ Owner only! 🔒",
            parse_mode='Markdown'
        )
        return

    action = args[0].lower()

    try:
        if action == "install":
            if len(args) < 2:
                await update.message.reply_text("❌ Specify package name")
                return
            package = args[1]
            await update.message.reply_text(f"📦 Installing {package}...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", package], capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                await update.message.reply_text(f"✅ Installed {package}")
            else:
                await update.message.reply_text(f"❌ Failed:\n```\n{result.stderr[:1000]}\n```", parse_mode='Markdown')

        elif action == "uninstall":
            if len(args) < 2:
                await update.message.reply_text("❌ Specify package name")
                return
            package = args[1]
            result = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", package], capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                await update.message.reply_text(f"✅ Uninstalled {package}")
            else:
                await update.message.reply_text(f"❌ Failed:\n```\n{result.stderr[:1000]}\n```", parse_mode='Markdown')

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
            "Fetch web content:\n"
            "• `/web https://example.com`\n"
            "• `/web https://api.example.com/data`\n\n"
            "Get any URL content! 🔗",
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
            await update.message.reply_text(f"🌐 **JSON Response:**\n```json\n{text}\n```", parse_mode='Markdown')
        elif 'html' in content_type:
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator='\n', strip=True)[:3500]
            await update.message.reply_text(f"🌐 **Page Content:**\n\n{text}")
        else:
            await update.message.reply_text(f"🌐 **Response ({response.status_code}):**\n```\n{response.text[:3500]}\n```", parse_mode='Markdown')

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)[:500]}")


async def math_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    expression = ' '.join(context.args) if context.args else None

    if not expression:
        await update.message.reply_text(
            "🔢 **Math Solver**\n\n"
            "Solve math problems:\n"
            "• `/math 2 + 2 * 10`\n"
            "• `/math sqrt(144)`\n"
            "• `/math solve x**2 - 4 = 0`\n"
            "• `/math diff x**2 + 3*x`\n"
            "• `/math integrate x**2`\n\n"
            "Advanced math support! 📐",
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

        elif expression.lower().startswith('simplify '):
            expr = sympify(expression[9:])
            result = simplify(expr)
            await update.message.reply_text(f"🔢 **Simplified:** `{result}` ✅", parse_mode='Markdown')

        elif expression.lower().startswith('expand '):
            expr = sympify(expression[7:])
            result = expand(expr)
            await update.message.reply_text(f"🔢 **Expanded:** `{result}` ✅", parse_mode='Markdown')

        elif expression.lower().startswith('factor '):
            expr = sympify(expression[7:])
            result = factor(expr)
            await update.message.reply_text(f"🔢 **Factored:** `{result}` ✅", parse_mode='Markdown')

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
            "Search the web:\n"
            "• `/search Python tutorials`\n"
            "• `/search latest news`\n"
            "• `/search weather today`\n\n"
            "Find anything online! 🌐",
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
            output = f"🔍 **Search Results for: {query}**\n\n" + "\n\n".join(results)
            await update.message.reply_text(output[:4000], parse_mode='Markdown')
        else:
            await update.message.reply_text("❌ No results found. Try different keywords.")

    except Exception as e:
        await update.message.reply_text(f"❌ Search error: {str(e)[:500]}")


async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    active_model = get_active_model(user_id)

    text = ' '.join(context.args) if context.args else None

    if not text:
        await update.message.reply_text(
            "🌍 **Translator**\n\n"
            "Translate text to any language:\n"
            "• `/translate to spanish: Hello world`\n"
            "• `/translate to japanese: Good morning`\n"
            "• `/translate to hindi: How are you?`\n\n"
            "Supports 100+ languages! 🗣️",
            parse_mode='Markdown'
        )
        return

    await update.message.reply_text("🌍 Translating... ⏳")

    prompt = f"Translate the following text. If no target language is specified, translate to English. Provide only the translation, nothing else:\n\n{text}"

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
            "Summarize any text or URL:\n"
            "• `/summarize https://example.com/article`\n"
            "• `/summarize [long text]`\n"
            "• Reply to a message with `/summarize`\n\n"
            "Get quick summaries! 📋",
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

    prompt = f"Provide a clear, concise summary of the following content. Include key points and main ideas:\n\n{content[:8000]}\n\nEnd with relevant emojis."

    try:
        result = await call_g4f(prompt, user_id)
        await update.message.reply_text(f"📝 **Summary:**\n\n{result}", parse_mode='Markdown')

    except Exception as e:
        await update.message.reply_text(f"❌ Summarize error: {str(e)[:500]}")


async def sysinfo_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        info = f"""
💻 **System Information**

🖥️ **Platform:** {platform.system()} {platform.release()}
🔧 **Architecture:** {platform.machine()}
🐍 **Python:** {platform.python_version()}
📁 **Working Dir:** {os.getcwd()}
"""

        if psutil:
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            info += f"""
⚡ **CPU Usage:** {cpu}%
🧠 **Memory:** {memory.percent}% ({memory.used // (1024**3)}GB / {memory.total // (1024**3)}GB)
💾 **Disk:** {disk.percent}% ({disk.used // (1024**3)}GB / {disk.total // (1024**3)}GB)
"""

        info += "\n✅ All systems operational! 🚀"

        await update.message.reply_text(info, parse_mode='Markdown')

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)[:500]}")


async def json_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = ' '.join(context.args) if context.args else None

    if not text and update.message.reply_to_message:
        text = update.message.reply_to_message.text

    if not text:
        await update.message.reply_text(
            "📋 **JSON Tool**\n\n"
            "Format and validate JSON:\n"
            "• `/json {\"name\": \"test\"}`\n"
            "• Reply to JSON with `/json`\n\n"
            "Pretty print JSON data! 📊",
            parse_mode='Markdown'
        )
        return

    try:
        data = json.loads(text)
        formatted = json.dumps(data, indent=2, ensure_ascii=False)
        await update.message.reply_text(f"📋 **Formatted JSON:**\n```json\n{formatted[:3500]}\n```\n✅ Valid JSON!", parse_mode='Markdown')
    except json.JSONDecodeError as e:
        await update.message.reply_text(f"❌ Invalid JSON:\n{str(e)}")


async def hash_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = ' '.join(context.args) if context.args else None

    if not text:
        await update.message.reply_text(
            "🔐 **Hash Generator**\n\n"
            "Generate various hashes:\n"
            "• `/hash hello world`\n\n"
            "Supports MD5, SHA1, SHA256, SHA512! 🔒",
            parse_mode='Markdown'
        )
        return

    try:
        md5 = hashlib.md5(text.encode()).hexdigest()
        sha1 = hashlib.sha1(text.encode()).hexdigest()
        sha256 = hashlib.sha256(text.encode()).hexdigest()
        sha512 = hashlib.sha512(text.encode()).hexdigest()

        result = f"""
🔐 **Hashes for:** `{text[:50]}`

**MD5:** `{md5}`
**SHA1:** `{sha1}`
**SHA256:** `{sha256}`
**SHA512:** `{sha512[:64]}...`

✅ Generated successfully! 🔒
"""
        await update.message.reply_text(result, parse_mode='Markdown')

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)[:500]}")


async def base64_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args if context.args else []

    if not args:
        await update.message.reply_text(
            "🔄 **Base64 Tool**\n\n"
            "Encode or decode base64:\n"
            "• `/base64 encode Hello World`\n"
            "• `/base64 decode SGVsbG8gV29ybGQ=`\n\n"
            "Convert data instantly! 🔐",
            parse_mode='Markdown'
        )
        return

    action = args[0].lower()
    text = ' '.join(args[1:])

    try:
        if action == "encode":
            result = base64.b64encode(text.encode()).decode()
            await update.message.reply_text(f"🔄 **Encoded:**\n`{result}`\n\n✅", parse_mode='Markdown')
        elif action == "decode":
            result = base64.b64decode(text.encode()).decode()
            await update.message.reply_text(f"🔄 **Decoded:**\n`{result}`\n\n✅", parse_mode='Markdown')
        else:
            await update.message.reply_text("❌ Use: `/base64 encode text` or `/base64 decode text`", parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)[:500]}")


async def regex_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = ' '.join(context.args) if context.args else None

    if not text or '|||' not in text:
        await update.message.reply_text(
            "🔤 **Regex Tester**\n\n"
            "Test regex patterns:\n"
            "• `/regex pattern ||| test string`\n"
            "• `/regex \\d+ ||| hello123world456`\n\n"
            "Find matches instantly! 🎯",
            parse_mode='Markdown'
        )
        return

    try:
        parts = text.split('|||')
        pattern = parts[0].strip()
        test_string = parts[1].strip()

        matches = re.findall(pattern, test_string)

        if matches:
            result = f"🔤 **Pattern:** `{pattern}`\n\n**Matches:** {matches[:20]}\n\n✅ Found {len(matches)} match(es)! 🎯"
        else:
            result = f"🔤 **Pattern:** `{pattern}`\n\n❌ No matches found."

        await update.message.reply_text(result, parse_mode='Markdown')

    except re.error as e:
        await update.message.reply_text(f"❌ Invalid regex: {str(e)}")
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
                "happy": "*khushi se jhoom uthi* Arey waah! Main bahut khush hoon ab! 😊✨",
                "excited": "*excitement se uchchhal gayi* YESSS! Main bahut excited hoon! Ye toh AMAZING hai! 🤩🎉",
                "loving": "*sharmaate hue muskurai* Aww, tum mujhe loving banna chahte ho? Kitne sweet ho tum! 🥰💕",
                "playful": "*shaitani smile deti hai* Ohoho! Ab masti ka time hai! 😜🎮",
                "frustrated": "*gehri saans leti hai* Theek hai... main frustrated hoon ab. *haath baandh liye* 😤",
                "angry": "*aankhen teekhi karti hai* Acha! Gussa chahiye? Lo dekho mera gussa! 😠💢",
                "sad": "*neeche dekhti hai* Oh... theek hai... main udaas hoon ab... *sniffles* 😢💔",
                "worried": "*hoth kaatne lagi* Arey... ab main worried feel kar rahi hoon... 😟",
                "curious": "*sir jhukate hue* Hmm! Ab main curious hoon! Aur batao! 🤔✨",
                "proud": "*seedha khadi hoti hai* Bilkul! Main proud feel kar rahi hoon! 😌👑",
                "neutral": "*shant andar se* Balanced aur steady. Samajh gayi. 🙂",
                "tired": "*ubaasi leti hai* Theek hai... thodi neend aa rahi hai... 😴💤",
                "flirty": "*aankh maarti hai* Ohho, flirty mood mein hoon ab! 😏💋",
                "grateful": "*dil pe haath rakh ke* Shukriya mere mood ka khayal rakhne ke liye! 🙏💖",
                "confident": "*confident smile* Oh haan! Main kuch bhi kar sakti hoon ab! 😎💪"
            }

            response = mood_reactions.get(requested_mood, f"*adjusts mood* I'm now feeling {requested_mood}! {mood_info['emoji']}")
            await update.message.reply_text(response)
        else:
            available_moods = ", ".join(MOODS.keys())
            await update.message.reply_text(
                f"🎭 **Available Moods:**\n\n{available_moods}\n\n"
                f"Use: `/mood happy` or `/mood angry` etc.",
                parse_mode='Markdown'
            )
    else:
        current = user_mood[user_id]
        mood_info = MOODS.get(current["mood"], MOODS["neutral"])
        history = current.get("history", [])[-5:]
        history_str = " → ".join([MOODS.get(m, MOODS["neutral"])["emoji"] for m in history]) if history else "No history"

        status = f"""
🎭 **Videl 🎀 Emotional State**

**Current Mood:** {current["mood"].upper()} {mood_info['emoji']}
**Feeling:** {mood_info['tone']}

**Recent Mood History:**
{history_str}

**Available Moods:**
😊 happy • 🤩 excited • 🥰 loving • 😜 playful
😤 frustrated • 😠 angry • 😢 sad • 😟 worried
🤔 curious • 😌 proud • 😴 tired • 😏 flirty
🙏 grateful • 😎 confident • 🙂 neutral

*{random.choice(mood_info['expressions'])}*

Use `/mood [mood]` to change my mood! 💫
"""
        await update.message.reply_text(status, parse_mode='Markdown')


async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [
            InlineKeyboardButton("💬 Chat", callback_data="menu_chat"),
            InlineKeyboardButton("🆓 Providers", callback_data="menu_providers")
        ],
        [
            InlineKeyboardButton("💻 Dev Tools", callback_data="menu_dev"),
            InlineKeyboardButton("🔧 Tools", callback_data="menu_tools")
        ],
        [
            InlineKeyboardButton("⚙️ Settings", callback_data="menu_settings"),
            InlineKeyboardButton("📊 Status", callback_data="menu_status")
        ],
        [
            InlineKeyboardButton("🎭 Mood", callback_data="menu_mood"),
            InlineKeyboardButton("❌ Close", callback_data="menu_close")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "🤖 **Videl 🎀 Control Panel**\n\n🆓 100% Free AI - No API Keys!\n\nSelect an option:",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    settings = user_settings[user_id]
    active_model = get_active_model(user_id)
    current_provider = user_g4f_provider.get(user_id, DEFAULT_G4F_PROVIDER)
    provider_info = G4F_PROVIDERS.get(current_provider, G4F_PROVIDERS[DEFAULT_G4F_PROVIDER])

    settings_text = f"""
⚙️ **Videl 🎀 Settings**

🤖 **Active Model:** {active_model.upper() if active_model else 'None'}
🔧 **Provider:** {provider_info['name']}
📝 **Detailed Responses:** {'ON' if settings['detailed_responses'] else 'OFF'}

**Commands:**
/model - Switch AI model
/providers - Change AI provider
/mood - Change bot mood
/clear - Clear conversation

🆓 All AI providers are FREE!
"""
    await update.message.reply_text(settings_text, parse_mode='Markdown')


GAMING_KEYWORDS = {
    "kill_words": ["maar", "maaro", "kill", "marna", "murder", "khatam", "finish", "end him", "end her", "attack"],
    "rob_words": ["rob", "loot", "chori", "steal", "chor", "looto", "paisa lelo", "money le"],
    "work_words": ["kaam", "work", "job", "naukri", "earning", "kamana", "paisa kamao"],
    "daily_words": ["daily", "reward", "claim", "bonus", "free money", "gift"],
    "heal_words": ["heal", "health", "treatment", "dawai", "medicine", "ilaj", "theek"],
    "game_words": ["game", "khel", "profile", "stats", "score", "rank"],
    "balance_words": ["balance", "paisa", "money", "wallet", "bank", "cash", "kitna hai"],
    "crime_words": ["crime", "criminal", "daaku", "robbery", "heist", "bank loot"],
    "revive_words": ["revive", "respawn", "alive", "zinda", "jaag", "uthao"],
    "leaderboard_words": ["leaderboard", "top", "ranking", "best players", "champions", "winners"],
    "challenge_words": ["challenge", "fight", "ladai", "duel", "pvp", "battle", "versus", "vs"],
    "taunt_words": ["noob", "weak", "kamzor", "loser", "gareeb", "poor", "chakka"],
}

GAMING_REACTIONS = {
    "kill_reaction": [
        "🎮 Arre kisi ko maarna hai? /kill use karo reply karke! ⚔️",
        "💀 Kill mode ON! /kill command use karo target ke message pe reply karke!",
        "🔫 Khatam karna hai? /kill likh ke reply karo! Maar dalo! 😈"
    ],
    "rob_reaction": [
        "💰 Looting time! /rob use karo kisi ke message pe reply karke! 🔫",
        "🏴‍☠️ Chor mode! /rob command try karo! Paisa loot lo! 💸",
        "😈 Rob karna hai? /rob likh ke reply karo victim ko!"
    ],
    "work_reaction": [
        "💼 Kaam karna hai? /work likhao aur paisa kamao! 💰",
        "👔 Job time! /work command se earning karo! 💵",
        "🛠️ Mehnat karo! /work use karo aur halal paisa lo! 💪"
    ],
    "daily_reaction": [
        "🎁 Daily reward lena hai? /daily likhao! Free paisa! 💰",
        "🎉 Free gift! /daily command se claim karo apna reward! 🎀",
        "💝 Roz ka inaam! /daily se lo apna bonus! ✨"
    ],
    "game_reaction": [
        "🎮 Game profile dekhna hai? /game likhao! 🏆",
        "📊 Apna stats check karo /game se! Kitne kill hain? 😎",
        "🎯 Gaming time! /game se apni profile dekho! ⚔️"
    ],
    "challenge_reaction": [
        "⚔️ Challenge accepted! /kill ya /rob use karo fight ke liye! 🔥",
        "🥊 Ladai chahiye? /kill command se maaro! Let's gooo! 💪",
        "🎯 PvP mode! Reply karo target ke message pe aur /kill ya /rob maro! 😈"
    ],
    "taunt_reaction": [
        "😏 Bahut bolte ho? Pehle apna /game profile to dekho! 🎮",
        "🤭 Arre bhai /bal check karo pehle! Kitna hai tere paas? 💰",
        "😂 Itna confidence? /lb dekho ranking! 🏆"
    ],
    "heal_reaction": [
        "💊 Heal chahiye? /heal use karo! Health recover ho jayegi! ❤️",
        "🏥 Doctor time! /heal command se apni health badhao! 💉",
        "❤️‍🩹 Injured ho? /heal likh ke theek ho jao! 🩺"
    ],
    "balance_reaction": [
        "💰 Paisa check karna hai? /bal likhao! 💵",
        "🏦 Bank balance? /bal se dekho kitna hai! 💸",
        "💵 Wallet check! /bal command use karo! 🤑"
    ],
    "crime_reaction": [
        "🔫 Crime time! /crime use karo risky paisa kamane ke liye! 💰",
        "🏴‍☠️ Daaku mode! /crime se bank loot! Risk hai par reward bhi! 😈",
        "💣 Criminal banna hai? /crime try karo! Police se bachna! 🚔"
    ],
    "revive_reaction": [
        "💀 Dead ho? /revive se wapas zinda ho jao! 🔄",
        "☠️ Respawn time! /revive likhao aur game mein wapas aao! ⚡",
        "🔄 Life back! /revive command se uthao apne aap ko! 💫"
    ],
    "leaderboard_reaction": [
        "🏆 Top players dekhne hain? /lb likhao! 🥇",
        "📊 Leaderboard check! /leaderboard se dekho kaun hai number 1! 🏅",
        "🥇 Champions list! /lb command se ranking dekho! 🌟"
    ]
}

async def detect_and_respond_gaming(update: Update, context: ContextTypes.DEFAULT_TYPE, message_lower: str, user_id: int) -> bool:
    chat_type = update.effective_chat.type
    if chat_type not in ["group", "supergroup"]:
        return False

    player = game_data[user_id]
    user = update.effective_user

    if update.message.reply_to_message:
        target_id = update.message.reply_to_message.from_user.id
        target_name = update.message.reply_to_message.from_user.first_name
        target = game_data[target_id]

        for word in GAMING_KEYWORDS["kill_words"]:
            if word in message_lower:
                if target_id == user_id:
                    await update.message.reply_text("🤦 Apne aap ko maar nahi sakta!")
                    return True

                if is_owner(target_id):
                    await update.message.reply_text("🛡️ Owner ko kill nahi kar sakta! Wo immortal hai!")
                    return True

                now = datetime.now()
                protect_until = target.get('protect_until')
                if protect_until and now < protect_until:
                    remaining = int((protect_until - now).total_seconds())
                    minutes = remaining // 60
                    await update.message.reply_text(f"🛡️ {target_name} protected hai!\n⏰ Protection ends in: {minutes}m")
                    return True

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
                    await update.message.reply_text(f"👤 {user.first_name} killed {target_name}!\n💰 Earned: ${loot}")
                else:
                    damage = random.randint(20, 40)
                    player['health'] = max(0, player['health'] - damage)
                    if player['health'] == 0:
                        player['status'] = 'dead'
                        player['deaths'] += 1
                        await update.message.reply_text(f"💀 {target_name} ne counter attack kiya!\n☠️ {user.first_name} DIED!")
                    else:
                        await update.message.reply_text(f"🛡️ {target_name} bach gaya!\n💔 You took {damage} damage!\n❤️ Health: {player['health']}%")
                return True

        for word in GAMING_KEYWORDS["rob_words"]:
            if word in message_lower:
                if target_id == user_id:
                    await update.message.reply_text("🤦 Apne aap ko rob nahi kar sakta!")
                    return True

                now = datetime.now()
                protect_until = target.get('protect_until')
                if protect_until and now < protect_until:
                    remaining = int((protect_until - now).total_seconds())
                    minutes = remaining // 60
                    await update.message.reply_text(f"🛡️ {target_name} protected hai!\n⏰ Protection ends in: {minutes}m")
                    return True

                if target['balance'] < 50:
                    await update.message.reply_text(f"😂 {target_name} ke paas kuch nahi hai! Gareeb hai!")
                    return True

                last_rob = player.get('last_rob')
                if last_rob:
                    time_diff = (now - last_rob).total_seconds()
                    if time_diff < GAME_COOLDOWNS['rob']:
                        remaining = int(GAME_COOLDOWNS['rob'] - time_diff)
                        minutes = remaining // 60
                        seconds = remaining % 60
                        await update.message.reply_text(f"⏰ Cooldown! Wait: {minutes}m {seconds}s")
                        return True

                player['last_rob'] = now
                success = random.random() > 0.5
                if success:
                    steal_amount = random.randint(int(target['balance'] * 0.1), int(target['balance'] * 0.3))
                    steal_amount = max(10, steal_amount)
                    player['balance'] += steal_amount
                    target['balance'] -= steal_amount
                    await update.message.reply_text(f"👤 {user.first_name} robbed ${steal_amount} from {target_name}!")
                else:
                    fine = random.randint(50, 150)
                    player['balance'] = max(0, player['balance'] - fine)
                    await update.message.reply_text(f"🚔 {target_name} ne police bulaya!\n💸 Fine: -${fine}\n💵 Balance: ${player['balance']}")
                return True

    for word in GAMING_KEYWORDS["challenge_words"]:
        if word in message_lower:
            response = random.choice(GAMING_REACTIONS["challenge_reaction"])
            await update.message.reply_text(response)
            return True

    for word in GAMING_KEYWORDS["taunt_words"]:
        if word in message_lower:
            response = random.choice(GAMING_REACTIONS["taunt_reaction"])
            await update.message.reply_text(response)
            return True

    for word in GAMING_KEYWORDS["kill_words"]:
        if word in message_lower:
            response = random.choice(GAMING_REACTIONS["kill_reaction"])
            await update.message.reply_text(response)
            return True

    for word in GAMING_KEYWORDS["rob_words"]:
        if word in message_lower:
            response = random.choice(GAMING_REACTIONS["rob_reaction"])
            await update.message.reply_text(response)
            return True

    for word in GAMING_KEYWORDS["work_words"]:
        if word in message_lower:
            response = random.choice(GAMING_REACTIONS["work_reaction"])
            await update.message.reply_text(response)
            return True

    for word in GAMING_KEYWORDS["daily_words"]:
        if word in message_lower:
            response = random.choice(GAMING_REACTIONS["daily_reaction"])
            await update.message.reply_text(response)
            return True

    for word in GAMING_KEYWORDS["game_words"]:
        if word in message_lower:
            response = random.choice(GAMING_REACTIONS["game_reaction"])
            await update.message.reply_text(response)
            return True

    for word in GAMING_KEYWORDS["heal_words"]:
        if word in message_lower:
            response = random.choice(GAMING_REACTIONS["heal_reaction"])
            await update.message.reply_text(response)
            return True

    for word in GAMING_KEYWORDS["balance_words"]:
        if word in message_lower:
            response = random.choice(GAMING_REACTIONS["balance_reaction"])
            await update.message.reply_text(response)
            return True

    for word in GAMING_KEYWORDS["crime_words"]:
        if word in message_lower:
            response = random.choice(GAMING_REACTIONS["crime_reaction"])
            await update.message.reply_text(response)
            return True

    for word in GAMING_KEYWORDS["revive_words"]:
        if word in message_lower:
            response = random.choice(GAMING_REACTIONS["revive_reaction"])
            await update.message.reply_text(response)
            return True

    for word in GAMING_KEYWORDS["leaderboard_words"]:
        if word in message_lower:
            response = random.choice(GAMING_REACTIONS["leaderboard_reaction"])
            await update.message.reply_text(response)
            return True

    return False


async def handle_gaming_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler that only responds to gaming-related keywords in messages"""
    if not update.message or not update.message.text:
        return
    user_id = update.effective_user.id
    message_lower = update.message.text.lower()
    await detect_and_respond_gaming(update, context, message_lower, user_id)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_message = update.message.text
    active_model = get_active_model(user_id)

    if not active_model:
        await update.message.reply_text(
            "No AI model is configured. This shouldn't happen - contact support."
        )
        return

    # Check for creator question
    message_lower = user_message.lower()
    for keyword in CREATOR_KEYWORDS:
        if keyword in message_lower:
            creator_response = """🥰😊

Mujhe mere bhagwan ne banaya hai Dev ji ne (@god_olds) 🙏✨

Woh mere creator hain, bahut talented developer hain! Unki wajah se main yahan hoon tumse baat karne ke liye! 💖🎀

Aur kuch jaanna hai mere baare mein? 😊✨"""
            await update.message.reply_text(creator_response)
            return

    # Gaming keyword detection for groups - auto respond to gaming related words
    gaming_response = await detect_and_respond_gaming(update, context, message_lower, user_id)
    if gaming_response:
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
        logger.error(f"AI response error: {e}")
        mood_error_responses = {
            "happy": "*smile fades a bit* Oh dear, I seem to be having a hiccup. Let me try again! 😅",
            "sad": "*sighs* I'm sorry... I couldn't process that. I feel terrible about it... 😢",
            "frustrated": "*grumbles* Ugh, technical difficulties! Give me a moment... 😤",
            "angry": "Blast! Something went wrong on my end. This is infuriating! 😠",
            "worried": "*looks concerned* Oh no, something's not right. I hope we can fix this... 😟",
            "playful": "*scratches head* Oops! I tripped over my own circuits there. Let's try again! 😜",
            "loving": "*looks apologetically* I'm so sorry, dear. Something went wrong but I'll fix it for you! 🥺",
            "neutral": "I apologize, I'm experiencing technical difficulties. Please try again. 🔄"
        }
        error_msg = mood_error_responses.get(new_mood, mood_error_responses["neutral"])
        await update.message.reply_text(error_msg)


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    data = query.data

    if data == "menu_close":
        await query.message.delete()

    elif data == "back_menu":
        keyboard = [
            [
                InlineKeyboardButton("💬 Chat", callback_data="menu_chat"),
                InlineKeyboardButton("🆓 Providers", callback_data="menu_providers")
            ],
            [
                InlineKeyboardButton("💻 Dev Tools", callback_data="menu_dev"),
                InlineKeyboardButton("🔧 Tools", callback_data="menu_tools")
            ],
            [
                InlineKeyboardButton("⚙️ Settings", callback_data="menu_settings"),
                InlineKeyboardButton("📊 Status", callback_data="menu_status")
            ],
            [
                InlineKeyboardButton("🎭 Mood", callback_data="menu_mood"),
                InlineKeyboardButton("❌ Close", callback_data="menu_close")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.edit_text(
            "🤖 **Videl 🎀 Control Panel**\n\n🆓 100% Free AI!\n\nSelect an option:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    elif data == "menu_chat":
        keyboard = [
            [
                InlineKeyboardButton("💬 Start Chat", callback_data="chat_start"),
                InlineKeyboardButton("🧹 Clear History", callback_data="chat_clear")
            ],
            [
                InlineKeyboardButton("💻 Code Help", callback_data="chat_code"),
                InlineKeyboardButton("🌍 Translate", callback_data="chat_translate")
            ],
            [
                InlineKeyboardButton("⬅️ Back", callback_data="back_menu"),
                InlineKeyboardButton("❌ Close", callback_data="menu_close")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.edit_text(
            "💬 **Chat Options**\n\nJust send any message to chat with me!\nNo commands needed - I'll respond with emotions and personality!",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    elif data == "menu_providers":
        current_provider = user_g4f_provider.get(user_id, DEFAULT_G4F_PROVIDER)
        keyboard = []
        for key, info in G4F_PROVIDERS.items():
            status = "✅ " if key == current_provider else ""
            keyboard.append([InlineKeyboardButton(f"{status}{info['name']}", callback_data=f"provider_{key}")])
        keyboard.append([
            InlineKeyboardButton("⬅️ Back", callback_data="back_menu"),
            InlineKeyboardButton("❌ Close", callback_data="menu_close")
        ])
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.edit_text(
            "🆓 **Free AI Providers**\n\nSelect a provider (all are FREE!):",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    elif data.startswith("provider_"):
        provider_key = data.replace("provider_", "")
        if provider_key in G4F_PROVIDERS:
            user_g4f_provider[user_id] = provider_key
            provider_info = G4F_PROVIDERS[provider_key]
            await query.message.edit_text(
                f"✅ Switched to **{provider_info['name']}**!\n\n"
                f"Models: {', '.join(provider_info['models'][:3])}\n\n"
                f"Just send me a message to start chatting! 🆓",
                parse_mode='Markdown'
            )

    elif data == "menu_dev":
        keyboard = [
            [
                InlineKeyboardButton("🐍 Run Python", callback_data="dev_python"),
                InlineKeyboardButton("🖥️ Shell", callback_data="dev_shell")
            ],
            [
                InlineKeyboardButton("📁 Files", callback_data="dev_files"),
                InlineKeyboardButton("📦 Pip", callback_data="dev_pip")
            ],
            [
                InlineKeyboardButton("⬅️ Back", callback_data="back_menu"),
                InlineKeyboardButton("❌ Close", callback_data="menu_close")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.edit_text(
            "💻 **Developer Tools**\n\nPowerful development environment:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    elif data == "menu_tools":
        keyboard = [
            [
                InlineKeyboardButton("🔍 Search", callback_data="tool_search"),
                InlineKeyboardButton("🌐 Web", callback_data="tool_web")
            ],
            [
                InlineKeyboardButton("🔢 Math", callback_data="tool_math"),
                InlineKeyboardButton("📝 Summarize", callback_data="tool_summarize")
            ],
            [
                InlineKeyboardButton("📋 JSON", callback_data="tool_json"),
                InlineKeyboardButton("🔐 Hash", callback_data="tool_hash")
            ],
            [
                InlineKeyboardButton("⬅️ Back", callback_data="back_menu"),
                InlineKeyboardButton("❌ Close", callback_data="menu_close")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.edit_text(
            "🔧 **Utility Tools**\n\nPowerful utilities at your fingertips:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    elif data == "menu_mood":
        current = user_mood[user_id]
        mood_info = MOODS.get(current["mood"], MOODS["neutral"])

        keyboard = [
            [
                InlineKeyboardButton("😊 Happy", callback_data="set_mood_happy"),
                InlineKeyboardButton("🤩 Excited", callback_data="set_mood_excited"),
                InlineKeyboardButton("🥰 Loving", callback_data="set_mood_loving")
            ],
            [
                InlineKeyboardButton("😜 Playful", callback_data="set_mood_playful"),
                InlineKeyboardButton("🤔 Curious", callback_data="set_mood_curious"),
                InlineKeyboardButton("😎 Confident", callback_data="set_mood_confident")
            ],
            [
                InlineKeyboardButton("😤 Frustrated", callback_data="set_mood_frustrated"),
                InlineKeyboardButton("😢 Sad", callback_data="set_mood_sad"),
                InlineKeyboardButton("😠 Angry", callback_data="set_mood_angry")
            ],
            [
                InlineKeyboardButton("⬅️ Back", callback_data="back_menu"),
                InlineKeyboardButton("❌ Close", callback_data="menu_close")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.edit_text(
            f"🎭 **Videl 🎀 Mood**\n\n"
            f"Current: **{current['mood'].upper()}** {mood_info['emoji']}\n"
            f"Feeling: {mood_info['tone']}\n\n"
            f"Select a mood to change how I feel:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    elif data.startswith("set_mood_"):
        new_mood = data.replace("set_mood_", "")
        if new_mood in MOODS:
            user_mood[user_id]["mood"] = new_mood
            user_mood[user_id]["history"].append(new_mood)
            mood_info = MOODS[new_mood]
            await query.message.edit_text(
                f"🎭 Mood changed to **{new_mood.upper()}** {mood_info['emoji']}\n\n"
                f"*{random.choice(mood_info['expressions'])}*",
                parse_mode='Markdown'
            )

    elif data == "menu_settings":
        settings = user_settings[user_id]
        active_model = get_active_model(user_id)
        current_provider = user_g4f_provider.get(user_id, DEFAULT_G4F_PROVIDER)
        provider_info = G4F_PROVIDERS.get(current_provider, G4F_PROVIDERS[DEFAULT_G4F_PROVIDER])

        keyboard = [
            [
                InlineKeyboardButton(
                    f"Detailed: {'ON' if settings['detailed_responses'] else 'OFF'}", 
                    callback_data="settings_detailed"
                )
            ],
            [
                InlineKeyboardButton("G4F", callback_data="model_g4f"),
                InlineKeyboardButton("FreeGPT", callback_data="model_freegpt")
            ],
            [
                InlineKeyboardButton("⬅️ Back", callback_data="back_menu"),
                InlineKeyboardButton("❌ Close", callback_data="menu_close")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.edit_text(
            f"⚙️ **Videl 🎀 Settings**\n\n"
            f"Active Model: **{active_model.upper() if active_model else 'None'}**\n"
            f"Provider: **{provider_info['name']}**\n"
            f"Detailed: **{'ON' if settings['detailed_responses'] else 'OFF'}**\n\n"
            f"🆓 All providers are FREE!",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    elif data == "menu_status":
        history_count = len(conversation_history[user_id])
        active_model = get_active_model(user_id)
        current_provider = user_g4f_provider.get(user_id, DEFAULT_G4F_PROVIDER)
        provider_info = G4F_PROVIDERS.get(current_provider, G4F_PROVIDERS[DEFAULT_G4F_PROVIDER])

        keyboard = [
            [
                InlineKeyboardButton("⬅️ Back", callback_data="back_menu"),
                InlineKeyboardButton("❌ Close", callback_data="menu_close")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.edit_text(
            f"📊 **Videl 🎀 Status**\n\n"
            f"🔌 Status: **Online**\n"
            f"🤖 Active AI: **{active_model.upper() if active_model else 'None'}**\n"
            f"🔧 Provider: **{provider_info['name']}**\n"
            f"💾 Memory: **{history_count} messages**\n"
            f"🎭 Mood: **{user_mood[user_id]['mood'].upper()}**\n\n"
            f"🆓 100% Free - No API Keys Required!",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    elif data.startswith("model_"):
        model = data.replace("model_", "")
        available = get_available_models()

        if model in available:
            user_ai_preference[user_id] = model
            await query.answer(f"Switched to {model.upper()}!")

    elif data == "settings_detailed":
        user_settings[user_id]["detailed_responses"] = not user_settings[user_id]["detailed_responses"]
        await query.answer(f"Detailed responses: {'ON' if user_settings[user_id]['detailed_responses'] else 'OFF'}")

    elif data == "chat_clear":
        conversation_history[user_id] = []
        await query.answer("Conversation cleared!")
        await query.message.edit_text("🧹 **Conversation Cleared!**\n\nStart a new chat by sending me a message.")

    elif data in ["chat_start", "chat_code", "chat_translate", "dev_python", "dev_shell", "dev_files", "dev_pip", 
                  "tool_search", "tool_web", "tool_math", "tool_summarize", "tool_json", "tool_hash"]:
        instructions = {
            "chat_start": "Just send me any message to chat!",
            "chat_code": "Use `/code your request` for coding help",
            "chat_translate": "Use `/translate to [language]: text`",
            "dev_python": "Use `/run your_python_code`",
            "dev_shell": "Use `/shell your_command`",
            "dev_files": "Use `/file list` or `/file read filename`",
            "dev_pip": "Use `/pip install package_name`",
            "tool_search": "Use `/search your query`",
            "tool_web": "Use `/web https://url.com`",
            "tool_math": "Use `/math expression`",
            "tool_summarize": "Use `/summarize text or URL`",
            "tool_json": "Use `/json {your: json}`",
            "tool_hash": "Use `/hash your text`"
        }
        await query.message.edit_text(f"ℹ️ {instructions.get(data, 'Feature coming soon!')}")


async def task_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    task_request = ' '.join(context.args) if context.args else None

    if not task_request:
        await update.message.reply_text(
            "🚀 **Autonomous Task Executor** 🎀\n\n"
            "Mujhe koi bhi complex task do, main khud steps mein break karke solve karungi!\n\n"
            "Examples:\n"
            "• `/task ek todo app ka code likh do`\n"
            "• `/task is code ko optimize karo`\n"
            "• `/task mujhe Python seekhna hai`\n"
            "• `/task ek story likho about AI`\n\n"
            "Main autonomous hoon - kuch bhi kar sakti hoon! 💪🎀",
            parse_mode='Markdown'
        )
        return

    await update.message.reply_text(f"🚀 *sooch rahi hoon* Task samajh gayi: `{task_request[:50]}...`\n\nRuk, main kaam karti hoon... ⏳", parse_mode='Markdown')

    prompt = f"""Tu ek autonomous AI agent hai. User ne ye task diya hai:

TASK: {task_request}

Apna approach Hinglish mein explain kar:
1. Pehle task ko samajh aur breakdown kar
2. Phir step by step solution de
3. Agar code chahiye toh likho with proper formatting
4. Agar explanation chahiye toh detail mein batao
5. End mein summary do

IMPORTANT: Response MUST be in Hinglish (Hindi words in English letters mixed with English). Be helpful, detailed, and friendly!"""

    try:
        current_mood_data = user_mood[user_id]
        new_mood, mood_reason = detect_mood_from_message(task_request, current_mood_data)
        dynamic_prompt = get_videl_prompt_with_mood(new_mood, mood_reason)

        result = await call_g4f(prompt, user_id, system_prompt=dynamic_prompt)

        if len(result) > 4000:
            for i in range(0, len(result), 4000):
                await update.message.reply_text(result[i:i+4000])
        else:
            await update.message.reply_text(result)

    except Exception as e:
        await update.message.reply_text(f"❌ Arey yaar, kuch gadbad ho gayi: {str(e)[:300]} 😅")


async def project_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    project_type = ' '.join(context.args) if context.args else None

    if not project_type:
        await update.message.reply_text(
            "📁 **Project Creator** 🎀\n\n"
            "Main tumhare liye projects bana sakti hoon!\n\n"
            "Examples:\n"
            "• `/project flask webapp`\n"
            "• `/project python script`\n"
            "• `/project calculator app`\n"
            "• `/project todo list`\n"
            "• `/project api server`\n\n"
            "Batao kya banana hai! 🚀",
            parse_mode='Markdown'
        )
        return

    await update.message.reply_text(f"📁 *excited ho gayi* Project bana rahi hoon: `{project_type}`... 🔨", parse_mode='Markdown')

    prompt = f"""Create a complete project structure and code for: {project_type}

Provide in Hinglish:
1. Project structure with files and folders
2. Complete working code for each file
3. Instructions on how to run it
4. Any dependencies needed

Format code in proper markdown code blocks with language specification.
Be detailed and make it production-ready!"""

    try:
        dynamic_prompt = get_videl_prompt_with_mood("excited", "Creating a project")

        result = await call_g4f(prompt, user_id, system_prompt=dynamic_prompt)

        if len(result) > 4000:
            for i in range(0, len(result), 4000):
                await update.message.reply_text(result[i:i+4000])
        else:
            await update.message.reply_text(result)

    except Exception as e:
        await update.message.reply_text(f"❌ Project create nahi ho paya: {str(e)[:300]}")


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    text = ' '.join(context.args) if context.args else None

    if not text and update.message.reply_to_message:
        text = update.message.reply_to_message.text

    if not text:
        await update.message.reply_text(
            "🔬 **Analyzer** 🎀\n\n"
            "Main kuch bhi analyze kar sakti hoon!\n\n"
            "Examples:\n"
            "• `/analyze [paste your code]`\n"
            "• `/analyze [any text]`\n"
            "• Reply to any message with `/analyze`\n\n"
            "Code bugs, security issues, improvements - sab bataungi! 🔍",
            parse_mode='Markdown'
        )
        return

    await update.message.reply_text("🔬 *dhyan se dekh rahi hoon* Analyzing... 🔍", parse_mode='Markdown')

    prompt = f"""Analyze the following content thoroughly:

{text[:4000]}

Provide detailed analysis in Hinglish including:
1. Kya hai ye (what is this)
2. Quality assessment
3. Agar code hai: bugs, security issues, improvements
4. Agar text hai: sentiment, key points, suggestions
5. Recommendations aur next steps

Be thorough but friendly!"""

    try:
        dynamic_prompt = get_videl_prompt_with_mood("curious", "Analyzing content")

        result = await call_g4f(prompt, user_id, system_prompt=dynamic_prompt)

        await update.message.reply_text(f"🔬 **Analysis Complete:**\n\n{result[:4000]}", parse_mode='Markdown')

    except Exception as e:
        await update.message.reply_text(f"❌ Analysis fail ho gayi: {str(e)[:300]}")


async def imagine_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    prompt_text = ' '.join(context.args) if context.args else None

    if not prompt_text:
        await update.message.reply_text(
            "✨ **Creative Imagination** 🎀\n\n"
            "Main creative content generate kar sakti hoon!\n\n"
            "Examples:\n"
            "• `/imagine ek love story Hindi mein`\n"
            "• `/imagine poem about nature`\n"
            "• `/imagine funny jokes`\n"
            "• `/imagine motivational speech`\n"
            "• `/imagine song lyrics`\n\n"
            "Batao kya imagine karun! 🌟",
            parse_mode='Markdown'
        )
        return

    await update.message.reply_text("✨ *creative mode on* Imagine kar rahi hoon... 🌈", parse_mode='Markdown')

    prompt = f"""Create this creative content: {prompt_text}

Make it:
1. Engaging aur interesting
2. High quality aur detailed
3. Hinglish mein (unless specifically asked for another language)
4. Emotionally impactful
5. Original aur unique

Let your creativity flow!"""

    try:
        dynamic_prompt = get_videl_prompt_with_mood("playful", "Creating something creative")

        result = await call_g4f(prompt, user_id, system_prompt=dynamic_prompt)

        if len(result) > 4000:
            for i in range(0, len(result), 4000):
                await update.message.reply_text(result[i:i+4000])
        else:
            await update.message.reply_text(f"✨ **Here you go:**\n\n{result}")

    except Exception as e:
        await update.message.reply_text(f"❌ Creative block ho gaya: {str(e)[:300]}")


async def debug_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    code = ' '.join(context.args) if context.args else None

    if not code and update.message.reply_to_message:
        code = update.message.reply_to_message.text

    if not code:
        await update.message.reply_text(
            "🐛 **Debug Master** 🎀\n\n"
            "Apna buggy code do, main fix kar dungi!\n\n"
            "Examples:\n"
            "• `/debug [paste your code with error]`\n"
            "• Reply to code with `/debug`\n\n"
            "Bugs ko dhund ke marungi! 🔨",
            parse_mode='Markdown'
        )
        return

    await update.message.reply_text("🐛 *detective mode* Bugs dhundh rahi hoon... 🔍", parse_mode='Markdown')

    prompt = f"""Debug this code and fix all issues:

```
{code[:3500]}
```

Provide in Hinglish:
1. Kya problems hain (list all bugs/issues)
2. Har bug ki explanation
3. FIXED code with proper formatting
4. Tips to avoid these bugs in future

Be thorough and educational!"""

    try:
        dynamic_prompt = get_videl_prompt_with_mood("confident", "Debugging code")

        result = await call_g4f(prompt, user_id, system_prompt=dynamic_prompt)

        if len(result) > 4000:
            for i in range(0, len(result), 4000):
                await update.message.reply_text(result[i:i+4000])
        else:
            await update.message.reply_text(result)

    except Exception as e:
        await update.message.reply_text(f"❌ Debug fail: {str(e)[:300]}")


async def explain_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    topic = ' '.join(context.args) if context.args else None

    if not topic and update.message.reply_to_message:
        topic = update.message.reply_to_message.text

    if not topic:
        await update.message.reply_text(
            "📚 **Explain Like I'm 5** 🎀\n\n"
            "Kuch bhi poocho, main simple mein samjhaungi!\n\n"
            "Examples:\n"
            "• `/explain quantum physics`\n"
            "• `/explain [paste complex code]`\n"
            "• `/explain machine learning`\n"
            "• `/explain blockchain`\n\n"
            "Koi bhi topic - main samjha dungi! 🧠",
            parse_mode='Markdown'
        )
        return

    await update.message.reply_text("📚 *teacher mode* Samjha rahi hoon... 📖", parse_mode='Markdown')

    prompt = f"""Explain this topic/code in simple terms that anyone can understand:

{topic[:3500]}

Requirements:
1. Use Hinglish (Hindi words in English letters + English mix)
2. Explain like teaching a beginner
3. Use simple analogies and examples
4. Break complex concepts into easy parts
5. Add relevant emojis to make it engaging
6. End with a summary

Make learning fun and easy!"""

    try:
        dynamic_prompt = get_videl_prompt_with_mood("curious", "Explaining a topic")

        result = await call_g4f(prompt, user_id, system_prompt=dynamic_prompt)

        if len(result) > 4000:
            for i in range(0, len(result), 4000):
                await update.message.reply_text(result[i:i+4000])
        else:
            await update.message.reply_text(result)

    except Exception as e:
        await update.message.reply_text(f"❌ Explain nahi ho paya: {str(e)[:300]}")


async def game_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show game profile with player stats"""
    user_id = update.effective_user.id
    user = update.effective_user

    # Get or create game data for user
    player = game_data[user_id]

    # Use username from Telegram if available, otherwise use default
    display_name = user.first_name if user.first_name else player["name"]

    game_profile = f"""🎮 **VIDEL GAME** 🎮

👤 Name: {display_name}
💰 Total Balance: ${player['balance']}
🏆 Global Rank: {player['rank']}
❤️ Status: {player['status']}
⚔️ Kills: {player['kills']}
💀 Deaths: {player['deaths']}
❤️‍🩹 Health: {player['health']}%

📋 **Game Commands:**
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
    """Check balance"""
    user_id = update.effective_user.id
    user = update.effective_user
    player = game_data[user_id]
    display_name = user.first_name if user.first_name else player["name"]

    if is_owner(user_id):
        bal_msg = f"""👑 **OWNER PROFILE** 👑
👤 Name: {display_name}
💰 Total Balance: ∞ (Unlimited)
🏆 Global Rank: #1 (Owner)
❤️ Status: IMMORTAL
⚔️ Kills: {player['kills']}
🛡️ Protection: PERMANENT"""
    else:
        bal_msg = f"""👤 Name: {display_name}
💰 Total Balance: ${player['balance']}
🏆 Global Rank: {player['rank']}
❤️ Status: {player['status']}
⚔️ Kills: {player['kills']}"""

    await update.message.reply_text(bal_msg, parse_mode='Markdown')


async def daily_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Claim daily reward"""
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
            await update.message.reply_text(f"⏰ Daily already claimed! Next in: {hours}h {minutes}m")
            return

    reward = random.randint(100, 500)
    player['balance'] += reward
    player['last_daily'] = now

    await update.message.reply_text(f"🎁 Daily reward claimed!\n💰 +${reward}\n💵 New Balance: ${player['balance']}")


async def work_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Work to earn money"""
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
            await update.message.reply_text(f"⏰ Abhi thak gaya hai! Next work in: {minutes}m {seconds}s")
            return

    jobs = ["programmer", "driver", "chef", "teacher", "doctor", "youtuber", "gamer"]
    job = random.choice(jobs)
    earnings = random.randint(50, 200)
    player['balance'] += earnings
    player['last_work'] = now

    await update.message.reply_text(f"💼 Tune {job} ki job ki!\n💰 +${earnings}\n💵 New Balance: ${player['balance']}")


async def crime_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Commit a crime (risky)"""
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
    success = random.random() > 0.4  # 60% success rate

    if success:
        loot = random.randint(200, 800)
        player['balance'] += loot
        crimes = ["bank robbery", "jewelry heist", "casino robbery", "car theft"]
        crime = random.choice(crimes)
        await update.message.reply_text(f"🔫 {crime.title()} successful!\n💰 +${loot}\n💵 Balance: ${player['balance']}")
    else:
        fine = random.randint(100, 300)
        player['balance'] = max(0, player['balance'] - fine)
        await update.message.reply_text(f"🚔 Police ne pakad liya!\n💸 Fine: -${fine}\n💵 Balance: ${player['balance']}")


async def rob_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Rob another user - supports /rob <amount> to rob specific coins"""
    user_id = update.effective_user.id
    user = update.effective_user
    player = game_data[user_id]
    player['name'] = user.first_name

    if not update.message.reply_to_message:
        await update.message.reply_text("⚠️ Kisi ke message pe reply karke rob kar!\nUsage: /rob or /rob <amount>")
        return

    target_id = update.message.reply_to_message.from_user.id
    target_name = update.message.reply_to_message.from_user.first_name

    if target_id == user_id:
        await update.message.reply_text("🤦 Apne aap ko rob nahi kar sakta!")
        return

    if is_owner(target_id):
        await update.message.reply_text("🛡️ Owner ko rob nahi kar sakta! Wo untouchable hai!")
        return

    target = game_data[target_id]
    target['name'] = target_name

    now = datetime.now()
    protect_until = target.get('protect_until')
    if protect_until and now < protect_until:
        remaining = int((protect_until - now).total_seconds())
        hours = remaining // 3600
        minutes = (remaining % 3600) // 60
        await update.message.reply_text(f"🛡️ {target_name} protected hai!\n⏰ Protection ends in: {hours}h {minutes}m")
        return

    if target['balance'] < 10:
        await update.message.reply_text(f"😂 {target_name} ke paas kuch nahi hai! Gareeb hai!")
        return

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

    requested_amount = None
    if context.args and len(context.args) > 0:
        try:
            requested_amount = int(context.args[0])
            if requested_amount <= 0:
                await update.message.reply_text("⚠️ Amount 0 se zyada hona chahiye!")
                return
        except ValueError:
            await update.message.reply_text("⚠️ Valid amount daal! Example: /rob 200")
            return

    success = random.random() > 0.5

    if success:
        if requested_amount and requested_amount > 0:
            steal_amount = min(requested_amount, target['balance'])
        else:
            steal_amount = random.randint(int(target['balance'] * 0.1), int(target['balance'] * 0.3))
        steal_amount = max(10, steal_amount)
        player['balance'] += steal_amount
        target['balance'] = max(0, target['balance'] - steal_amount)
        status_msg = " (💀 Dead)" if target['status'] == 'dead' else ""
        await update.message.reply_text(f"👤 {user.first_name} robbed ${steal_amount} from {target_name}{status_msg}!\n💵 Your Balance: ${player['balance']}")
    else:
        fine = random.randint(50, 150)
        player['balance'] = max(0, player['balance'] - fine)
        await update.message.reply_text(f"🚔 {target_name} ne police bulaya!\n💸 Fine: -${fine}\n💵 Balance: ${player['balance']}")


async def kill_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Kill another user"""
    user_id = update.effective_user.id
    user = update.effective_user
    player = game_data[user_id]
    player['name'] = user.first_name

    if not update.message.reply_to_message:
        await update.message.reply_text("⚠️ Kisi ke message pe reply karke kill kar!")
        return

    target_id = update.message.reply_to_message.from_user.id
    target_name = update.message.reply_to_message.from_user.first_name

    if target_id == user_id:
        await update.message.reply_text("🤦 Apne aap ko kill nahi kar sakta!")
        return

    if is_owner(target_id):
        await update.message.reply_text("🛡️ Owner ko kill nahi kar sakta! Wo immortal hai!")
        return

    target = game_data[target_id]
    target['name'] = target_name

    now = datetime.now()
    protect_until = target.get('protect_until')
    if protect_until and now < protect_until:
        remaining = int((protect_until - now).total_seconds())
        minutes = remaining // 60
        await update.message.reply_text(f"🛡️ {target_name} protected hai!\n⏰ Protection ends in: {minutes}m")
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

        await update.message.reply_text(f"👤 {user.first_name} killed {target_name}!\n💰 Earned: ${loot}")
    else:
        damage = random.randint(20, 40)
        player['health'] = max(0, player['health'] - damage)
        if player['health'] == 0:
            player['status'] = 'dead'
            player['deaths'] += 1
            await update.message.reply_text(f"💀 {target_name} ne counter attack kiya!\n☠️ {user.first_name} DIED!")
        else:
            await update.message.reply_text(f"🛡️ {target_name} bach gaya!\n💔 You took {damage} damage!\n❤️ Health: {player['health']}%")


async def heal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Heal yourself"""
    user_id = update.effective_user.id
    player = game_data[user_id]

    if player['status'] == 'dead':
        await update.message.reply_text("💀 Tu dead hai! Pehle /revive kar!")
        return

    if player['health'] >= 100:
        await update.message.reply_text("❤️ Health already full hai!")
        return

    cost = 50
    if player['balance'] < cost:
        await update.message.reply_text(f"💸 Not enough money! Need ${cost} to heal!")
        return

    player['balance'] -= cost
    heal_amount = random.randint(20, 50)
    player['health'] = min(100, player['health'] + heal_amount)

    await update.message.reply_text(f"💊 Healed!\n❤️ +{heal_amount} HP\n❤️ Health: {player['health']}%\n💵 Balance: ${player['balance']}")


async def revive_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Revive another player - you cannot revive yourself!"""
    user_id = update.effective_user.id
    user = update.effective_user
    player = game_data[user_id]
    player['name'] = user.first_name

    if player['status'] == 'dead':
        await update.message.reply_text("💀 Tu khud dead hai! Kisi aur se apni revive karwa!\n⚠️ Tu apne aap ko revive nahi kar sakta!")
        return

    if not update.message.reply_to_message:
        await update.message.reply_text("⚠️ Kisi DEAD player ke message pe reply karke revive kar!\n💸 Cost: $500\n⚠️ Note: Tu apne aap ko revive nahi kar sakta!")
        return

    target_id = update.message.reply_to_message.from_user.id
    target_name = update.message.reply_to_message.from_user.first_name

    if target_id == user_id:
        await update.message.reply_text("🤦 Apne aap ko revive nahi kar sakta!\n⚠️ Kisi aur se apni revive karwa!")
        return

    target = game_data[target_id]
    target['name'] = target_name

    if target['status'] != 'dead':
        await update.message.reply_text(f"❤️ {target_name} already alive hai!")
        return

    if player['balance'] < REVIVE_COST and not is_owner(user_id):
        await update.message.reply_text(f"💸 Not enough money! Need ${REVIVE_COST} to revive {target_name}!\n💵 Your balance: ${player['balance']}")
        return

    if not is_owner(user_id):
        player['balance'] -= REVIVE_COST

    target['status'] = 'alive'
    target['health'] = 100

    await update.message.reply_text(f"🔄 {user.first_name} ne {target_name} ko revive kar diya!\n💸 Cost: ${REVIVE_COST}\n❤️ {target_name} Status: ALIVE\n❤️ Health: 100%\n💵 Your Balance: ${player['balance']}")


async def protect_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Buy 24 hour protection from kills and robs for $500"""
    user_id = update.effective_user.id
    player = game_data[user_id]

    if player['status'] == 'dead':
        await update.message.reply_text("💀 Tu dead hai! Pehle kisi se /revive karwa!")
        return

    now = datetime.now()
    protect_until = player.get('protect_until')

    if protect_until and now < protect_until:
        remaining = int((protect_until - now).total_seconds())
        hours = remaining // 3600
        minutes = (remaining % 3600) // 60
        await update.message.reply_text(f"🛡️ Tu already protected hai!\n⏰ Time left: {hours}h {minutes}m")
        return

    if player['balance'] < PROTECT_COST and not is_owner(user_id):
        await update.message.reply_text(f"💸 Not enough money! Need ${PROTECT_COST} for 24h protection!\n💵 Your balance: ${player['balance']}")
        return

    if not is_owner(user_id):
        player['balance'] -= PROTECT_COST

    player['protected'] = True
    player['protect_until'] = now + timedelta(seconds=GAME_COOLDOWNS['protect'])

    await update.message.reply_text(f"🛡️ Protection activated!\n⏰ Duration: 24 hours\n💸 Cost: ${PROTECT_COST}\n💵 Balance: ${player['balance']}\n\n🔒 Nobody can kill or rob you for 24 hours!")


async def give_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Give money to another user (10% tax)"""
    user_id = update.effective_user.id
    player = game_data[user_id]

    if player['status'] == 'dead':
        await update.message.reply_text("💀 Tu dead hai! Pehle /revive kar!")
        return

    if not update.message.reply_to_message:
        await update.message.reply_text("⚠️ Kisi ke message pe reply karke give kar!\nUsage: /give <amount>")
        return

    if not context.args:
        await update.message.reply_text("⚠️ Amount bata! Example: /give 500")
        return

    try:
        amount = int(context.args[0])
    except ValueError:
        await update.message.reply_text("⚠️ Valid number daal! Example: /give 500")
        return

    if amount < 10:
        await update.message.reply_text("⚠️ Minimum $10 give kar sakta hai!")
        return

    tax = int(amount * 0.1)
    total_cost = amount + tax

    if player['balance'] < total_cost:
        await update.message.reply_text(f"💸 Not enough money!\n💰 Amount: ${amount}\n📊 Tax (10%): ${tax}\n💵 Total needed: ${total_cost}\n💵 Your balance: ${player['balance']}")
        return

    target_id = update.message.reply_to_message.from_user.id
    target_name = update.message.reply_to_message.from_user.first_name

    if target_id == user_id:
        await update.message.reply_text("🤦 Apne aap ko paise nahi de sakta!")
        return

    target = game_data[target_id]

    player['balance'] -= total_cost
    target['balance'] += amount

    await update.message.reply_text(f"✅ You gave ${amount} to {target_name} with ${tax} fee deducted! (10% tax applied) 💸")


async def leaderboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show leaderboard - ranked by kills and money, starting from rank 1000"""
    if not game_data:
        await update.message.reply_text("📊 No players yet!")
        return

    sorted_players = sorted(
        game_data.items(), 
        key=lambda x: (x[1]['kills'] * 1000 + x[1]['balance']), 
        reverse=True
    )[:10]

    lb_text = "🏆 **LEADERBOARD** 🏆\n"
    lb_text += "━━━━━━━━━━━━━━━━━━\n\n"
    medals = ["🥇", "🥈", "🥉"]

    base_rank = 1000
    for i, (uid, data) in enumerate(sorted_players):
        medal = medals[i] if i < 3 else f"#{base_rank - i}"
        if i < 3:
            rank_display = f"{medal} #{base_rank - i}"
        else:
            rank_display = f"#{base_rank - i}"
        
        status_emoji = "❤️" if data['status'] == 'alive' else "💀"
        name = data.get('name', 'Unknown')[:12]
        
        if is_owner(uid):
            lb_text += f"{rank_display} 👑 **{name}**\n   💰 ∞ | ⚔️{data['kills']} kills | {status_emoji}\n\n"
        else:
            lb_text += f"{rank_display} **{name}**\n   💰 ${data['balance']} | ⚔️{data['kills']} kills | {status_emoji}\n\n"

    lb_text += "━━━━━━━━━━━━━━━━━━\n"
    lb_text += "📊 Ranked by: Kills + Money"

    await update.message.reply_text(lb_text, parse_mode='Markdown')


async def welcome_new_members(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome new members when they join a group"""
    for member in update.message.new_chat_members:
        if member.is_bot:
            continue

        welcome_msg = f"""🎀 **Welcome to the group, {member.first_name}!** 🎀

Hey {member.first_name}! 🤗💖

Main hoon **Videl** - is group ki AI dost! 

🎮 **Game Commands:**
/game - Apna profile dekho
/daily - Daily reward lo
/work - Kaam karke paisa kamao
/kill - Kisi ko maaro (reply karke)
/rob - Kisi ko looto

💬 **Chat:** Sirf message karo, main jawab dungi!
📸 **Photo:** Photo bhejo, main analyze karungi!

Enjoy karo aur masti karo! 🎀✨"""

        await update.message.reply_text(welcome_msg, parse_mode='Markdown')


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Analyze photos sent by users"""
    user_id = update.effective_user.id

    await update.message.reply_text("🔍 Photo analyze kar rahi hoon... 📸")

    try:
        # Get the largest photo
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)

        # Download photo
        photo_bytes = await file.download_as_bytearray()

        # Convert to base64 for API
        photo_base64 = base64.b64encode(photo_bytes).decode('utf-8')

        # Get caption if any
        caption = update.message.caption or "Is photo mein kya hai? Describe in detail."

        # Use g4f with vision capable model
        try:
            from g4f.client import Client
            from g4f.Provider import Blackbox

            client = Client()

            # Create image data URL
            image_url = f"data:image/jpeg;base64,{photo_base64}"

            response = client.chat.completions.create(
                model="gpt-4o",
                provider=Blackbox,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""Tu Videl hai 🎀 - ek friendly AI jo Hinglish mein baat karti hai.
Is photo ko analyze kar aur batao:
1. Photo mein kya dikhai de raha hai?
2. Koi special details?
3. Interesting observations?

User ka question: {caption}

Hinglish mein jawab de, friendly aur detailed!"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            }
                        ]
                    }
                ]
            )

            analysis = response.choices[0].message.content

            if len(analysis) > 4000:
                for i in range(0, len(analysis), 4000):
                    await update.message.reply_text(analysis[i:i+4000])
            else:
                await update.message.reply_text(f"📸 **Photo Analysis** 🎀\n\n{analysis}")

        except Exception as e:
            logger.error(f"Vision API error: {e}")
            # Fallback response
            await update.message.reply_text(
                "📸 Photo mil gayi! 🎀\n\n"
                "Abhi vision feature thoda busy hai, but maine photo receive kar li!\n"
                "Thodi der mein try karo ya caption ke saath photo bhejo! 💖✨"
            )

    except Exception as e:
        logger.error(f"Photo handling error: {e}")
        await update.message.reply_text(
            "😅 Photo process karne mein thodi problem hui!\n"
            "Please dubara try karo! 🎀"
        )


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
    application.add_handler(CommandHandler("protect", protect_command))
    application.add_handler(CommandHandler("give", give_command))
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
    application.add_handler(CommandHandler("json", json_command))
    application.add_handler(CommandHandler("hash", hash_command))
    application.add_handler(CommandHandler("base64", base64_command))
    application.add_handler(CommandHandler("regex", regex_command))
    application.add_handler(CommandHandler("mood", mood_command))
    application.add_handler(CommandHandler("menu", menu_command))
    application.add_handler(CommandHandler("settings", settings_command))
    application.add_handler(CommandHandler("task", task_command))
    application.add_handler(CommandHandler("project", project_command))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CommandHandler("imagine", imagine_command))
    application.add_handler(CommandHandler("debug", debug_command))
    application.add_handler(CommandHandler("explain", explain_command))

    application.add_handler(CallbackQueryHandler(callback_handler))

    application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, welcome_new_members))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_gaming_message))

    logger.info("Videl 🎀 Free AI Bot started! 🚀")
    print("🤖 Videl 🎀 Free AI Bot is running!")
    print("🆓 Using 100% free AI providers - No API keys required!")
    print("✨ NEW: Addy ChatGPT API and Gemini API added!")

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
