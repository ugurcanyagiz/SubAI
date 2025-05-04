
import os
import whisper
import srt
import yt_dlp
import openai
import streamlit as st
import subprocess
from datetime import timedelta
from dotenv import load_dotenv

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

LANGUAGES = {
    "English": "en", "Turkish": "tr", "French": "fr", "Spanish": "es",
    "German": "de", "Italian": "it", "Russian": "ru", "Korean": "ko",
    "Japanese": "ja", "Chinese (Simplified)": "zh", "Arabic": "ar",
    "Portuguese": "pt", "Hindi": "hi", "Dutch": "nl"
}

st.set_page_config(page_title="SubAI Web Translator", layout="centered")
st.title("🎬 SubAI – Subtitle Translator")

url = st.text_input("YouTube Link")
target_lang_label = st.selectbox("Target Language", list(LANGUAGES.keys()), index=0)
run_translate = st.button("Translate")

if run_translate and url:
    with st.spinner("Downloading video..."):
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            'merge_output_format': 'mp4',
            'outtmpl': 'downloads/%(id)s.%(ext)s',
            'quiet': True
        }
        os.makedirs("downloads", exist_ok=True)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info['id']
            video_path = f"downloads/{video_id}.mp4"
            audio_path = f"downloads/{video_id}_audio.mp3"

    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "mp3", audio_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    st.info("Transcribing audio with Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    segments = result['segments']

    target_lang_code = LANGUAGES[target_lang_label]

    def translate_text(text):
        prompt = f"Translate the following subtitle to {target_lang_label}:\n\n{text}"
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": (
                        f"You are an expert subtitle translator. Translate the user's input to {target_lang_label}. "
                        "Do not censor, tone down, or soften expressions. Preserve slang, sarcasm, insults, emotional tension, and cultural context. "
                        "Translate as a native speaker would say it in real life—casual, raw, and emotionally honest. "
                        "Use natural language, not robotic or overly formal tone. "
                        "If an expression has multiple meanings, choose the one that fits best with the emotional and narrative context of the entire video. "
                        "Output only the translated subtitle text."
                    )},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return text

    st.info("Translating subtitles with GPT-4...")
    subtitles = []
    for i, seg in enumerate(segments):
        start = timedelta(seconds=seg['start'])
        end = timedelta(seconds=seg['end'])
        translated = translate_text(seg['text'].strip())
        subtitles.append(srt.Subtitle(index=i+1, start=start, end=end, content=translated))

    srt_data = srt.compose(subtitles)
    srt_path = f"downloads/{video_id}_{target_lang_code}.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_data)

    with open(srt_path, "rb") as file:
        st.success("Translation complete! Download your subtitle:")
        st.download_button(label="📥 Download SRT File", data=file, file_name=os.path.basename(srt_path), mime="text/plain")
