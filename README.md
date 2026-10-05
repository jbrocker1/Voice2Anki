# Voice2Anki

Voice2Anki turns voice recordings into high-quality [Anki](https://apps.ankiweb.net/) flashcards. Record what you want to memorize, and Alfred (the assistant) corrects the Whisper transcript and ships formatted cards straight to Anki.

**[Features](#features) · [Gallery](#gallery) · [Installation](#installation) · [Usage](#usage) · [Launch options](#launch-options) · [Troubleshooting](#troubleshooting) · [FAQ](#faq)**

## Features

- **Any language, any subject** — works on any Whisper transcript
- **Adaptive (Few-Shot) learning** — your corrections become examples for future cards
- **Multiple LLMs** — OpenAI, Replicate, OpenRouter, anything [litellm](https://docs.litellm.ai/) supports
- **Multiple TTS engines** for audio output
- **Custom formatting** — drop a Python function in `flashcard_editor.py` to control card output
- **Smart OCR** — image-based notes get indexed for Anki search
- **Profiles** — separate settings, decks, and example banks per use case

## Gallery

- ![First](./docs/first.png)
- ![Second](./docs/second.png)
- ![Third](./docs/third.png)
- ![Fourth](./docs/fourth.png)
- ![Fifth](./docs/fifth.png)
- ![Sixth](./docs/sixth.png)
- ![Seventh](./docs/seventh.png)

## Installation

### Prerequisites

- **Python 3.8+** on `PATH` (the installer pins the app environment to Python 3.13)
- **[Anki](https://apps.ankiweb.net/)** with the **[AnkiConnect](https://ankiweb.net/shared/info/2055492159)** add-on installed and enabled
- **An OpenAI API key** (or any other LLM provider supported by litellm)

Everything else is handled automatically: `ffmpeg`/`ffprobe` are bundled via `static-ffmpeg`, no admin rights needed. The installer is idempotent — re-run after a `git pull` to pick up dependency changes, add `--recreate` to rebuild from scratch, or run `python setup_env.py --check` for a component status report.

### Windows

1. Double-click **`setup.bat`** — it builds the environment and asks to launch.
2. Later, just double-click **`run.bat`** to start the app.

### macOS / Linux

```bash
git clone https://github.com/thiswillbeyourgithub/Voice2Anki.git
cd Voice2Anki
python setup_env.py
source .venv/bin/activate
python Voice2Anki.py
```

### Optional: Tesseract (OCR mode only)

Tesseract is a separate C++ binary and the one thing that cannot be pip-installed. You only need it for the OCR feature — audio, transcription, and Anki import all work without it.

- **Windows**: `winget install UB-Mannheim.TesseractOCR`
- **macOS**: `brew install tesseract`
- **Debian / Ubuntu**: `sudo apt install tesseract-ocr`
- **Fedora**: `sudo dnf install tesseract`

### Prefer your own venv?

`pip install -r requirements.txt` works, but you then need to install `ffmpeg` and `ffprobe` yourself and put them on `PATH` before importing pydub.

## Usage

### Start the app

1. Open Anki (AnkiConnect must be running)
2. Launch Voice2Anki:
   - **Windows**: double-click `run.bat`
   - **Any platform**: from the repo root, `.venv\Scripts\python.exe Voice2Anki.py` (Windows) / `.venv/bin/python Voice2Anki.py` (macOS / Linux)
3. Your browser opens automatically. The console also prints the URL — by default `http://127.0.0.1:7860`.

### First-time setup in the UI

1. Pick a profile name in the `Profile` field — this loads default settings for that profile
2. Enter your API key in the settings
3. Pick a model in the `LLM` dropdown on the `Memories & Buffer` tab — the default entry is a placeholder and won't generate cards
4. Fill the `LLM context` box on the `Controls` tab (e.g. "I'm learning Japanese; make cloze cards from what I record") — Alfred refuses to generate cards without it
5. Set a `Deck name` and `Tags` on the `Anki` tab — `Ankify` refuses to send cards without them

### Daily workflow

1. Hit record, speak your flashcard content
2. Stop — Alfred returns a corrected cloze
3. Edit if needed, then `Ankify` to push to Anki
4. *(optional)* Save the transcript as an example to teach Alfred your style

## Launch options

By default Voice2Anki runs **only on localhost** (`http://127.0.0.1:7860`), with **no login prompt**, and opens your browser for you.

| Flag | Effect |
| --- | --- |
| `--port 8000` | Serve on a different port |
| `--localnetwork` | Also expose to your LAN (`http://[LOCAL_IP]:7860`). Plain HTTP — trusted network only |
| `--authentication` | Require the login/password pairs hardcoded in `Voice2Anki.py` (default `v2a` / `v2a` — change them there). Auto-on with `--share` |
| `--open_browser=False` | Don't auto-open the browser |
| `--debug` | Verbose logging for troubleshooting |
| `--share` | Temporary public URL (72h) via Hugging Face — **use with caution** |

To turn a flag **off**, pass `--name=False`. `--no-name` does **not** work and will either crash or be silently ignored.

Run `python Voice2Anki.py --help` for the full list. On Windows, flags pass through `run.bat`: e.g. `run.bat --port 8000`.

## Tips & Tricks

- **Add hints at the end of recordings** to steer Alfred:
  - "Note to Alfred: 3 cards on this topic"
  - "Note to Alfred: list card"
- **Save good transcripts as examples** — remove the hint first, then keep them as Few-Shot prompts for future cards
- **Profiles** keep example prompts and decks separate per subject/card style

## Troubleshooting

### Recordings come out silent

Windows' own mic test can pass while the browser records silence, because the browser may be capturing a different or virtual input:

- **Windows privacy gate**: Settings → Privacy & security → Microphone → enable **"Let desktop apps access your microphone"**. The master toggle alone is not enough — browsers count as desktop apps, and the OS mic test bypasses this gate.
- **Wrong input device**: in Chrome check `chrome://settings/content/microphone` (Edge: `edge://settings/content/microphone`) and select your *real* mic. Machines with Voicemeeter, NVIDIA Broadcast, or OBS expose several silent virtual inputs and don't always default to the real one.
- Each audio slot in Voice2Anki also has a **"Select input device"** dropdown to pick the mic per recording.
- **Quick isolation test**: open any browser-based mic test page in the same browser — if it's silent there too, the problem is browser/OS-level, not Voice2Anki.

### AnkiConnect is unreachable

- Make sure Anki is running
- AnkiConnect add-on must be installed and enabled (default port `8765`)
- If you see `sync: auth not configured`, Voice2Anki prints a warning and skips syncing — card creation still works

### Something else

- **Verbose logs**: run with `--debug`
- **Component status**: `python setup_env.py --check`
- **Still stuck**: open an issue with the traceback and your OS / Python version

## Important notes

- **SSL**: only relevant with `--localnetwork`. Drop a cert pair into `utils/ssl/` (`key.pem` and `cert.pem`) and Voice2Anki uses HTTPS; otherwise plain HTTP.
- **Browser performance**: Chromium-based browsers give better CPU performance than Firefox.
- **Updates**: `git pull`, then re-run `setup_env.py` to apply any dependency changes.
- **Syncing**: optional. Set your AnkiWeb login in Anki's `Tools > Preferences > Syncing` to enable.

## FAQ

<details>
<summary>Why was this project created?</summary>

Voice2Anki started as a tool for end-of-medical-school flashcards, then was released and documented with [aider](https://aider.chat/). It worked well for me and should be made much more accessible.

</details>

<details>
<summary>How did you use it?</summary>

Record long audio sessions speaking out cards separated by an audible "STOP", split them automatically with [Whisper Audio Splitter](https://github.com/thiswillbeyourgithub/whisper_audio_splitter), then batch-process with Voice2Anki. Very efficient for creating large numbers of high-quality cards.

</details>

<details>
<summary>What are profiles and why should I use them?</summary>

Profiles separate example prompts. For instance, a "medical lists" profile and a "physics" profile keep their own Few-Shot banks and decks so cards don't get contaminated by an off-context example.

</details>

<details>
<summary>What does OCR have to do with Voice2Anki?</summary>

[OCR_with_format](https://github.com/thiswillbeyourgithub/OCR_with_format) runs on screenshots from textbooks so the native Anki search browser also searches image content.

</details>

<details>
<summary>What is Few-Shot Learning?</summary>

A machine-learning approach where a model adapts from a small number of examples. When you save a corrected card as an example, Voice2Anki retrieves the closest past examples and feeds them to the LLM so future cards match your phrasing and style. More: [geeksforgeeks](https://www.geeksforgeeks.org/zero-shot-vs-one-shot-vs-few-shot-learning/#what-is-fewshot-learning).

</details>

<details>
<summary>Do I have to use the Gradio GUI?</summary>

No — there's a script at `utils/cli.py` that runs autonomously. It's experimental, barely tested, and should be used with caution.

</details>

<details>
<summary>How can I help?</summary>

Help wanted especially on Python packaging into Anki and making installation easier via PyPI. Documentation improvements are also welcome.

</details>

## Support

- Open one on GitHub for bugs / feature requests
- Feedback is always welcome and helps the project grow