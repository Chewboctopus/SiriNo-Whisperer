# MacWhisperer 🎙️

Welcome to **MacWhisperer**! This is my own personal implementation and flavor of OpenAI's Whisper speech-to-text neural network, specifically tailored and optimized for Mac users.

It runs entirely locally on your Mac's Apple Silicon GPU using the MLX framework, meaning it requires **zero internet connection**, has **zero API costs**, and fully respects your **privacy**.

### 🤝 Open Source & Forking
Because this is my own personal version, you are completely free to fork it, change it, and adapt it however you like for your own workflows! If you have ideas or suggestions for improvements, feel free to open a Pull Request or an Issue. I'd love to see what you build with it!

---

## ✨ Features
* **Global Hotkey:** Press `Right Option` from inside *any* application to instantly start dictating.
* **Auto-Pasting:** When you press `Enter`, the AI seamlessly warps you back to your original app and instantly types out everything you said.
* **Live Editable Preview:** See a live preview of the AI's transcription in real-time. If you spot a mistake, you can manually type and edit the grey preview text before committing it!
* **Contextual AI Brain:** Uses the advanced `whisper-base` model. Unlike traditional speech-to-text, it evaluates the context of your entire paragraph mathematically to resolve tricky homophones (e.g. "band" vs "banned").
* **Custom Jargon Dictionary:** Add your industry's specific software, proper nouns, and jargon to a custom dictionary to mathematically force the AI to spell them correctly.
* **VAD Gate:** Built-in Voice Activity Detection physically blocks the AI when you aren't speaking, completely eliminating "static hallucinations" and text-doubling bugs.

## 🚀 Installation

### Prerequisites
* A Mac with Apple Silicon (M1, M2, M3, M4)
* Python 3.10 or higher

### 1. Automated Installation (Recommended)
Simply double-click the `install.command` file from Finder! 
This script will automatically:
1. Check for and install Homebrew (if you don't have it)
2. Install the necessary audio system libraries (`ffmpeg`, `portaudio`)
3. Create a secure Python virtual environment
4. Download all the required AI libraries

### 2. Permissions
MacWhisperer uses AppleScript and `pynput` to securely navigate between your apps. 
The first time you run it, macOS will ask you to grant **Accessibility Permissions** to your Terminal application (System Settings -> Privacy & Security -> Accessibility).

## 🎮 How It Works

To start the background listener, simply double-click the `start_whisper.command` file! It will run silently in the background waiting for you.

**The Workflow:**
1. **Focus:** Click into *any* app where you want to type (e.g. Chrome, Mail, VS Code, Notes).
2. **Trigger:** Tap the **Right Option** key on your keyboard and start speaking.
3. **The Floating Window:** The moment you start speaking, a semi-transparent floating UI window will appear at the bottom of your screen. As you speak, you will see a live preview of the AI transcribing your voice in real-time.
4. **Pause & Edit:** If you need to pause to gather your thoughts, just tap the **Right Command** key. You can actually click directly into the floating window and use your keyboard to manually fix any typos or edit the text mid-sentence!
5. **Multi-Tasking:** If you need to read an email on another screen while dictating, simply click out of the MacWhisperer window. It will keep listening. When you are done reading, click the floating window to re-focus it.
6. **Paste:** When you are finished, press **Enter**. The floating window will vanish, warp you back to the exact app you were originally typing in, and instantly paste your text!

## ⚙️ Customization
Open `main.py` to customize the internal AI engine:
* **`HOTKEY`**: Change the trigger key (default is `Right Option`).
* **`JARGON`**: Add your custom names, software, or industry jargon here separated by commas.
* **`REPLACEMENTS`**: Add custom word replacements to completely ban filler words (like "um") or auto-correct specific misheard phrases.
* **`MODEL`**: If you want even higher accuracy, upgrade to `mlx-community/whisper-large-v3-turbo`!

## 🙏 Attributions 
This project was built by Brian Wankum in collaboration with [Antigravity](https://github.com/google-antigravity), an autonomous AI agent by Google DeepMind. 

It stands on the shoulders of these incredible open-source giants:
* [OpenAI Whisper](https://github.com/openai/whisper): The core speech-to-text neural network.
* [Apple MLX](https://github.com/ml-explore/mlx): The framework that allows Whisper to run at blistering speeds on Apple Silicon GPUs. 
* [Hugging Face](https://huggingface.co): For hosting the open-source community MLX models.
