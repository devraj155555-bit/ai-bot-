import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

# Load environment variables from .env file (for local/external hosting)
load_dotenv()

# API Keys & URLs
AI_INTEGRATIONS_OPENAI_API_KEY = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
AI_INTEGRATIONS_OPENAI_BASE_URL = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")

AI_INTEGRATIONS_ANTHROPIC_API_KEY = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_API_KEY")
AI_INTEGRATIONS_ANTHROPIC_BASE_URL = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_BASE_URL")

AI_INTEGRATIONS_GEMINI_API_KEY = os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY")
AI_INTEGRATIONS_GEMINI_BASE_URL = os.environ.get("AI_INTEGRATIONS_GEMINI_BASE_URL")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Model Names
OPENAI_MODEL = "gpt-5"
ANTHROPIC_MODEL = "claude-sonnet-4-5"
GEMINI_MODEL = "gemini-3-pro-preview"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Initialize clients conditionally
openai_client = None
if AI_INTEGRATIONS_OPENAI_API_KEY:
    from openai import OpenAI
    openai_client = OpenAI(
        api_key=AI_INTEGRATIONS_OPENAI_API_KEY,
        base_url=AI_INTEGRATIONS_OPENAI_BASE_URL or "https://api.openai.com/v1"
    )
    print("✅ OpenAI client initialized")
else:
    print("⚠️  OpenAI key not set, GPT-5 unavailable")

anthropic_client = None
if AI_INTEGRATIONS_ANTHROPIC_API_KEY:
    from anthropic import Anthropic
    anthropic_client = Anthropic(
        api_key=AI_INTEGRATIONS_ANTHROPIC_API_KEY,
        base_url=AI_INTEGRATIONS_ANTHROPIC_BASE_URL or "https://api.anthropic.com"
    )
    print("✅ Anthropic client initialized")
else:
    print("⚠️  Anthropic key not set, Claude unavailable")

gemini_client = None
if AI_INTEGRATIONS_GEMINI_API_KEY:
    from google import genai
    gemini_client = genai.Client(
        api_key=AI_INTEGRATIONS_GEMINI_API_KEY,
        http_options={
            'api_version': '',
            'base_url': AI_INTEGRATIONS_GEMINI_BASE_URL or "https://generativelanguage.googleapis.com"
        }
    )
    print("✅ Gemini client initialized")
else:
    print("⚠️  Gemini key not set, Gemini unavailable")

groq_client = None
if GROQ_API_KEY:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("✅ Groq (backup) client initialized")
else:
    print("⚠️  Groq key not set, backup brain unavailable")


def is_rate_limit_error(exception: BaseException) -> bool:
    error_msg = str(exception)
    return (
        "429" in error_msg
        or "RATELIMIT_EXCEEDED" in error_msg
        or "quota" in error_msg.lower()
        or "rate limit" in error_msg.lower()
        or (hasattr(exception, "status_code") and getattr(exception, "status_code", None) == 429)
        or (hasattr(exception, "status") and getattr(exception, "status", None) == 429)
    )

def get_backup_response(prompt: str) -> str:
    if not groq_client:
        return "❌ Backup brain (Groq) not configured. Please set GROQ_API_KEY."
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a backup AI coding assistant. Provide high-quality code snippets."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content or "Backup brain returned empty response."
    except Exception as e:
        return f"Backup brain error: {str(e)}"

@retry(
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=1, min=2, max=128),
    retry=retry_if_exception(is_rate_limit_error),
    reraise=True
)
def get_gpt5_response(prompt: str) -> str:
    if not openai_client:
        raise RuntimeError("OpenAI not configured. Set AI_INTEGRATIONS_OPENAI_API_KEY.")
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert AI coding agent powered by GPT-5. Always provide high-quality, production-ready code snippets."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=8192
    )
    return response.choices[0].message.content or "Error with GPT-5"

@retry(
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=1, min=2, max=128),
    retry=retry_if_exception(is_rate_limit_error),
    reraise=True
)
def get_claude_response(prompt: str) -> str:
    if not anthropic_client:
        raise RuntimeError("Anthropic not configured. Set AI_INTEGRATIONS_ANTHROPIC_API_KEY.")
    message = anthropic_client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=8192,
        system="You are an expert at code logic and debugging powered by Claude. Focus on fixing errors and improving structural integrity.",
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text if message.content[0].type == "text" else "Error with Claude"

@retry(
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=1, min=2, max=128),
    retry=retry_if_exception(is_rate_limit_error),
    reraise=True
)
def get_gemini_response(prompt: str) -> str:
    if not gemini_client:
        raise RuntimeError("Gemini not configured. Set AI_INTEGRATIONS_GEMINI_API_KEY.")
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={"system_instruction": "You are a design and UI expert. Provide beautiful CSS, UI components, and design guidance."}
    )
    return response.text or "Error with Gemini"


# Set up Discord bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    if bot.user:
        print(f'✅ Logged in as {bot.user.name} (ID: {bot.user.id})')
        print('------')

@bot.command(name="code")
async def code(ctx, *, prompt: str):
    """Routes to the best AI model based on prompt keywords."""
    async with ctx.typing():
        try:
            p_lower = prompt.lower()

            if any(k in p_lower for k in ["design", "ui", "css", "html", "style", "frontend"]):
                try:
                    response = get_gemini_response(prompt)
                    provider = "Gemini 3"
                except Exception as e:
                    if "FREE_CLOUD_BUDGET_EXCEEDED" in str(e) or "not configured" in str(e).lower():
                        response = get_backup_response(prompt)
                        provider = "Llama-3 (Backup)"
                    else:
                        raise e

            elif any(k in p_lower for k in ["fix", "error", "bug", "logic", "reasoning", "refactor"]):
                try:
                    response = get_claude_response(prompt)
                    provider = "Claude Sonnet"
                except Exception as e:
                    if "FREE_CLOUD_BUDGET_EXCEEDED" in str(e) or "not configured" in str(e).lower():
                        response = get_backup_response(prompt)
                        provider = "Llama-3 (Backup)"
                    else:
                        raise e

            else:
                try:
                    response = get_gpt5_response(prompt)
                    provider = "GPT-5"
                except Exception as e:
                    if "FREE_CLOUD_BUDGET_EXCEEDED" in str(e) or "not configured" in str(e).lower():
                        response = get_backup_response(prompt)
                        provider = "Llama-3 (Backup)"
                    else:
                        raise e

            full_response = f"**[AI: {provider}]**\n{response}"

            if len(full_response) > 2000:
                for i in range(0, len(full_response), 2000):
                    await ctx.send(full_response[i:i+2000])
            else:
                await ctx.send(full_response)

        except Exception as e:
            await ctx.send(f"❌ An error occurred: {str(e)}")
            print(f"Error: {e}")

if __name__ == "__main__":
    token = os.environ.get("DISCORD_TOKEN")
    if token:
        bot.run(token)
    else:
        print("❌ Error: DISCORD_TOKEN not set.")
