import os
import re
import gc
import sys
import time
import zlib
import queue
import signal
import tempfile
import threading
import subprocess

import numpy as np
import sounddevice as sd
import soundfile as sf
import mlx_whisper
import dotenv
import tkinter as tk
from pynput import keyboard
from pynput.mouse import Controller as MouseController, Button

# Load secure environment variables
dotenv.load_dotenv()

# --- First-Time Setup Window ---
def check_setup():
    if not os.environ.get("HF_TOKEN"):
        root = tk.Tk()
        root.title("SiriNo Whisperer Setup")
        root.geometry("500x200")
        root.attributes('-topmost', True)
        
        tk.Label(root, text="Welcome to SiriNo Whisperer! 🎙️", font=("Helvetica", 16, "bold")).pack(pady=10)
        tk.Label(root, text="To download the local AI model, please enter your Hugging Face Token:").pack()
        
        entry = tk.Entry(root, width=50)
        entry.pack(pady=10)
        
        def save():
            token = entry.get().strip()
            if token:
                with open(".env", "a") as f:
                    f.write(f"\nHF_TOKEN={token}\n")
                os.environ["HF_TOKEN"] = token
                root.destroy()
                
        tk.Button(root, text="Save & Continue", command=save).pack()
        root.mainloop()

check_setup()

# --- Single Instance Lock ---
def kill_previous_instance():
    pid_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sirino.pid")
    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as f:
                old_pid = int(f.read().strip())
            
            # Verify the PID actually belongs to SiriNo Whisperer before killing it
            cmd = subprocess.run(
                ['ps', '-p', str(old_pid), '-o', 'command='],
                capture_output=True, text=True
            ).stdout
            if "python" in cmd.lower() and "main.py" in cmd.lower():
                os.kill(old_pid, signal.SIGKILL)
                print(f"💀 Killed old background process (PID {old_pid}) to ensure a fresh start.")
        except Exception:
            pass
    try:
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))
    except Exception:
        pass

kill_previous_instance()

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
MODE = os.environ.get("WHISPER_MODE", "local").lower()
HOTKEY = 'Right Option (Toggle)'

# mlx-community/whisper-tiny-mlx   = Absolute fastest, lowest accuracy
# mlx-community/whisper-base-mlx   = Excellent speed, much better accuracy
# mlx-community/whisper-small-mlx  = Balanced: good accuracy, moderate speed
# mlx-community/whisper-large-v3-turbo = State of the art, but requires more RAM
MODEL = "mlx-community/whisper-large-v3-turbo"
SAMPLE_RATE = 16000
CHUNK_INTERVAL = 0.5  # How often (seconds) the dispatch loop checks for new audio

# Tuning constants
MAX_TRANSCRIBE_SECONDS = 10   # Never send more than this to MLX (prevents GPU stalls)
AUTO_COMMIT_SECONDS = 8       # Bake text into prefix before buffer hits the cap
VAD_SILENCE_THRESHOLD = 0.015 # Amplitude below this = silence (Mac mic noise floor ~0.002)
VAD_TAIL_SECONDS = 0.5        # Only check the most recent audio for silence (not the whole buffer)
VAD_WIPE_SECONDS = 5          # Wipe buffer after this much continuous silence
MAX_TRANSCRIBE_SAMPLES = SAMPLE_RATE * MAX_TRANSCRIBE_SECONDS
AUTO_COMMIT_SAMPLES = SAMPLE_RATE * AUTO_COMMIT_SECONDS
VAD_TAIL_SAMPLES = int(SAMPLE_RATE * VAD_TAIL_SECONDS)
VAD_WIPE_SAMPLES = SAMPLE_RATE * VAD_WIPE_SECONDS
MIN_AUDIO_SAMPLES = int(SAMPLE_RATE * 0.3)  # Need at least 0.3s of audio to transcribe

# Custom Jargon Dictionary
JARGON = ""
jargon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jargon.txt")
if os.path.exists(jargon_path):
    with open(jargon_path, "r") as f:
        JARGON = f.read().strip()

# Word Replacements (Banned Words & Auto-Correct)
REPLACEMENTS = {
    " gonna ": " going to ",
    " gonna.": " going to.",
    " um ": " ",
    " uh ": " ",
    "free pick": "Freepik",
    "free pic": "Freepik",
    "serino": "SiriNo",
    "cyrano": "SiriNo",
    "Cimes": "seems",
    "cimes": "seems"
}

# Pre-compile replacement patterns once (not inside the hot loop)
COMPILED_REPLACEMENTS = [
    (re.compile(re.escape(bad), re.IGNORECASE), good)
    for bad, good in REPLACEMENTS.items()
]


# ──────────────────────────────────────────────────────────────────────
# Application
# ──────────────────────────────────────────────────────────────────────
class SiriNoWhispererApp:
    def __init__(self):
        # --- State ---
        self.recording = False
        self.is_window_open = False
        self.session_id = 0
        self.text_prefix = ""
        self.text_suffix = ""
        self.target_app = ""
        self.breadcrumb_pos = (0, 0)
        self.focus_lost_to_ui = False
        self.last_alt_r_time = 0
        self.last_transcribe_time = 0

        # --- Thread-safe audio buffer ---
        self.audio_data = []
        self.audio_lock = threading.Lock()

        # --- Transcription worker (single persistent thread) ---
        self.work_queue = queue.Queue(maxsize=1)
        self._worker_thread = threading.Thread(target=self._transcription_worker, daemon=True)

        # Initialize CoreGraphics Controllers on the MAIN thread to prevent macOS BPT Trap 5 crashes
        self.keyboard_controller = keyboard.Controller()
        self.mouse_controller = MouseController()
        
        # --- UI Setup ---
        self.root = tk.Tk()
        
        # Hide the Python Rocket Icon from the Dock and Cmd-Tab menu
        try:
            from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
            app = NSApplication.sharedApplication()
            app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        except Exception:
            pass
            
        self.root.title("SiriNo Whisperer Preview")
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.95)
        self.root.configure(bg='#1e1e1e')
        
        # Position at bottom center
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = 800
        window_height = 150
        x = (screen_width - window_width) // 2
        y = screen_height - window_height - 100
        self.root.geometry(f'{window_width}x{window_height}+{x}+{y}')
        
        # Editable Text Widget
        self.text_widget = tk.Text(
            self.root, 
            fg='#ffffff', 
            bg='#1e1e1e',
            font=("Helvetica", 20),
            wrap=tk.WORD,
            insertbackground='white',
            padx=15,
            pady=15,
            borderwidth=0,
            highlightthickness=0
        )
        self.text_widget.pack(expand=True, fill='both')
        
        # Bindings
        self.text_widget.bind("<Return>", self.on_enter)
        self.text_widget.bind("<Escape>", self.on_escape)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close_window)
        
        # Thread-safe UI update queue
        self.ui_queue = queue.Queue()
        self.root.after(50, self._process_ui_queue)
        
        # Hide initially
        self.root.withdraw()
        
        # --- Audio ---
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, callback=self._audio_callback
        )
        self.stream.start()
        
        # --- Start threads ---
        self._worker_thread.start()
        
        self._dispatch_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._dispatch_thread.start()
        
        self.listener = keyboard.Listener(on_release=self._on_key_release)
        self.listener.start()

    # ──────────────────────────────────────────────────────────────────
    # UI Queue (runs on Tkinter main thread)
    # ──────────────────────────────────────────────────────────────────
    def _process_ui_queue(self):
        try:
            while True:
                func = self.ui_queue.get_nowait()
                func()
        except queue.Empty:
            pass
        self.root.after(50, self._process_ui_queue)

    # ──────────────────────────────────────────────────────────────────
    # Audio Callback (runs on sounddevice's C thread — must be fast)
    # ──────────────────────────────────────────────────────────────────
    def _audio_callback(self, indata, frames, time_info, status):
        if self.recording:
            with self.audio_lock:
                self.audio_data.append(indata.copy())

    # ──────────────────────────────────────────────────────────────────
    # Hotkey Handler (runs on pynput's listener thread)
    # ──────────────────────────────────────────────────────────────────
    def _on_key_release(self, key):
        if key == keyboard.Key.alt_r:
            # Debounce: prevent software-emulated key releases from double-launching
            now = time.time()
            if now - self.last_alt_r_time < 0.5:
                return
            self.last_alt_r_time = now
            
            if not self.is_window_open:
                self.ui_queue.put(self.start_recording)
            elif not self.recording:
                # Edit Mode (Paused) → Resume
                self.ui_queue.put(self.start_recording)
            else:
                # Live Mode → Submit
                self.ui_queue.put(self.submit_and_paste)
                
        elif key == keyboard.Key.cmd_r:
            if self.is_window_open:
                if self.recording:
                    self.ui_queue.put(self.stop_recording)
                else:
                    self.ui_queue.put(self.start_recording)

    # ──────────────────────────────────────────────────────────────────
    # Recording State Machine
    # ──────────────────────────────────────────────────────────────────
    def play_sound(self, sound_name):
        try:
            subprocess.Popen(["afplay", f"/System/Library/Sounds/{sound_name}.aiff"])
        except Exception:
            pass

    def start_recording(self):
        print("🎤 Recording started...")
        self.play_sound("Ping")
        self.recording = True
        self.root.title("SiriNo Whisperer - 🔴 LIVE (Tap Right Option to Submit)")
        
        if self.is_window_open:
            # Resuming a paused session — split text at cursor position
            # so new transcription is inserted exactly where the user left their cursor.
            cursor_pos = self.text_widget.index(tk.INSERT)
            before_cursor = self.text_widget.get("1.0", cursor_pos)
            after_cursor = self.text_widget.get(cursor_pos, "end-1c")
            
            with self.audio_lock:
                if before_cursor.strip() and before_cursor.strip() != "Listening...":
                    # Add trailing space so live text doesn't smash into prefix
                    self.text_prefix = before_cursor.rstrip() + " "
                else:
                    self.text_prefix = ""
                if after_cursor.strip():
                    # Add leading space so suffix doesn't smash into live text
                    self.text_suffix = " " + after_cursor.lstrip()
                else:
                    self.text_suffix = ""
        else:
            # Fresh session
            self.session_id += 1
            with self.audio_lock:
                self.text_prefix = ""
                self.text_suffix = ""
            self.focus_lost_to_ui = False
            
            # Drop a Mouse Coordinate Breadcrumb
            try:
                self.breadcrumb_pos = self.mouse_controller.position
            except Exception:
                pass
                
            self.text_widget.delete("1.0", tk.END)
            self.text_widget.insert(tk.END, "Listening...")
            
            self.root.attributes('-topmost', True)
            self.is_window_open = True
            self.root.deiconify()
            self.root.lift()
            self.root.update()
            
            # Force Python to take focus
            try:
                subprocess.Popen([
                    'osascript', '-e',
                    'tell application "System Events" to set frontmost of process "Python" to true'
                ])
            except Exception:
                pass
            self.root.focus_force()
            self.text_widget.focus_set()
            
        # Wipe audio buffer for fresh start
        with self.audio_lock:
            self.audio_data = []

    def stop_recording(self):
        print("⏸️ Recording paused (Editable).")
        self.play_sound("Pop")
        self.recording = False
        self.root.title("SiriNo Whisperer - ⏸️ PAUSED (Edit your text, then press Enter to Paste)")

    def on_enter(self, event):
        self.submit_and_paste()
        return "break"

    def on_escape(self, event):
        self._cancel_session()
        return "break"

    def on_close_window(self):
        self._cancel_session()

    def _cancel_session(self):
        self.recording = False
        self.is_window_open = False
        with self.audio_lock:
            self.text_prefix = ""
            self.text_suffix = ""
            self.audio_data = []
        self.root.withdraw()

    def submit_and_paste(self):
        self.recording = False
        self.is_window_open = False
        
        final_text = self.text_widget.get("1.0", "end-1c").strip()
        self.text_widget.delete("1.0", tk.END)
        self.root.withdraw()
        
        with self.audio_lock:
            self.text_prefix = ""
            self.text_suffix = ""
            self.audio_data = []
        
        if final_text and final_text != "Listening...":
            threading.Thread(target=self._paste_text_mac, args=(final_text,), daemon=True).start()

    # ──────────────────────────────────────────────────────────────────
    # Dispatch Loop (background thread — prepares audio, never blocks)
    # ──────────────────────────────────────────────────────────────────
    def _dispatch_loop(self):
        """Checks for new audio and dispatches work to the persistent worker thread.
        If the worker is still busy, the work item is silently dropped (queue maxsize=1).
        This guarantees the loop NEVER blocks regardless of how slow MLX is."""
        while True:
            try:
                if self.recording and MODE == "local":
                    now = time.time()
                    if now - self.last_transcribe_time > CHUNK_INTERVAL:
                        self.last_transcribe_time = now
                        
                        # Peek at buffer for VAD check (lightweight, no copy)
                        with self.audio_lock:
                            if len(self.audio_data) == 0:
                                time.sleep(0.1)
                                continue
                            audio_peek = np.concatenate(self.audio_data, axis=0)
                        
                        if len(audio_peek) < MIN_AUDIO_SAMPLES:
                            time.sleep(0.1)
                            continue
                            
                        # --- VAD Gate ---
                        if np.max(np.abs(audio_peek)) < VAD_SILENCE_THRESHOLD:
                            if len(audio_peek) > VAD_WIPE_SAMPLES:
                                with self.audio_lock:
                                    self.audio_data = []
                            time.sleep(0.1)
                            continue
                        
                        # Signal the worker: "audio is ready, come get it"
                        # We send just the session_id — the worker takes its own
                        # fresh snapshot to avoid stale-data duplication bugs.
                        try:
                            try:
                                self.work_queue.get_nowait()  # Replace stale signal
                            except queue.Empty:
                                pass
                            self.work_queue.put_nowait(self.session_id)
                        except queue.Full:
                            pass
            except Exception as e:
                print(f"⚠️ Dispatch loop error (recovering): {e}")
                                
            time.sleep(0.1)

    # ──────────────────────────────────────────────────────────────────
    # Transcription Worker (single persistent background thread)
    # ──────────────────────────────────────────────────────────────────
    def _transcription_worker(self):
        """Persistent thread that takes FRESH buffer snapshots at processing time.
        This eliminates stale-data bugs where the dispatch copies audio before an
        auto-commit wipe, causing the worker to re-transcribe already-committed text."""
        while True:
            try:
                # Block until signaled (no CPU spin)
                session_id = self.work_queue.get()
                
                # Stale session check
                if session_id != self.session_id:
                    continue
                
                # Take FRESH buffer snapshot (not stale dispatch-time data!)
                with self.audio_lock:
                    if len(self.audio_data) == 0:
                        continue
                    audio_copy = self.audio_data.copy()
                
                audio_np = np.concatenate(audio_copy, axis=0)
                buffer_length = len(audio_np)
                
                if buffer_length < MIN_AUDIO_SAMPLES:
                    continue
                
                # Buffer cap
                if buffer_length > MAX_TRANSCRIBE_SAMPLES:
                    audio_to_transcribe = audio_np[-MAX_TRANSCRIBE_SAMPLES:]
                else:
                    audio_to_transcribe = audio_np
                    
                text = self._transcribe_chunk(audio_to_transcribe)
                
                # Stale session check after transcription
                if session_id != self.session_id:
                    continue
                
                # --- Auto-Commit ---
                if buffer_length > AUTO_COMMIT_SAMPLES and text:
                    with self.audio_lock:
                        self.text_prefix += text + " "
                        self.audio_data = []
                    # Drain any stale signals that reference pre-wipe state
                    try:
                        self.work_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.ui_queue.put(lambda sid=session_id: self._update_text_widget("", sid))
                else:
                    self.ui_queue.put(lambda t=text, sid=session_id: self._update_text_widget(t, sid))
                    
            except Exception as e:
                print(f"⚠️ Transcription worker error (recovering): {e}")

    # ──────────────────────────────────────────────────────────────────
    # Transcription Engine
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _is_hallucination(text):
        lower = text.lower()
        if "thank you for watching" in lower or "thanks for watching" in lower:
            return True
        if text.strip() in ("You", "I"):
            return True
        encoded = text.encode('utf-8')
        if len(encoded) > 30 and (len(zlib.compress(encoded)) / len(encoded)) < 0.35:
            print("🧹 Suppressed repetitive Whisper hallucination!")
            return True
        return False

    def _transcribe_chunk(self, audio_np):
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)
        try:
            sf.write(temp_path, audio_np, SAMPLE_RATE)
            
            result = mlx_whisper.transcribe(
                temp_path, 
                path_or_hf_repo=MODEL, 
                condition_on_previous_text=False,
                initial_prompt=JARGON
            )
            text = result.get('text', '').strip()
            if not text:
                return ""
                
            if self._is_hallucination(text):
                return ""
            
            # Apply pre-compiled word replacements (no per-call re.compile overhead)
            for pattern, replacement in COMPILED_REPLACEMENTS:
                text = pattern.sub(replacement, text)
            
            # --- De-Shouting Filter ---
            if text.isupper() and len(text) > 4:
                text = text.capitalize()
            else:
                words = text.split(" ", 1)
                first_alpha = "".join(c for c in words[0] if c.isalpha())
                if first_alpha.isupper() and len(first_alpha) > 4:
                    words[0] = words[0].capitalize()
                    text = " ".join(words)
                    
            # Ensure sentence capitalization
            if text:
                text = text[0].upper() + text[1:]
            
            return text
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    # ──────────────────────────────────────────────────────────────────
    # UI Updates (called on Tkinter main thread via ui_queue)
    # ──────────────────────────────────────────────────────────────────
    def _update_text_widget(self, text, session_id):
        if self.session_id != session_id:
            return
            
        with self.audio_lock:
            prefix = self.text_prefix
            suffix = self.text_suffix
        
        full_text = prefix + text + suffix
        
        # Calculate cursor target: end of the live text (between prefix and suffix)
        cursor_char_offset = len(prefix) + len(text)
        
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert(tk.END, full_text)
        
        # Place cursor at the end of the live transcription zone,
        # NOT at the very end of the widget (which would be after the suffix)
        if suffix:
            self.text_widget.mark_set(tk.INSERT, f"1.0 + {cursor_char_offset} chars")
            self.text_widget.see(tk.INSERT)
        else:
            self.text_widget.see(tk.END)
        
        # Crash backup
        try:
            backup_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backup.txt")
            with open(backup_path, "w") as f:
                f.write(full_text)
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────
    # Paste Macro (runs in its own thread to avoid blocking UI)
    # ──────────────────────────────────────────────────────────────────
    def _execute_mouse_click(self):
        self.mouse_controller.position = self.breadcrumb_pos
        self.mouse_controller.click(Button.left)

    def _paste_text_mac(self, text):
        # 1. Set clipboard
        process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
        process.communicate(text.encode('utf-8'))
        
        # 2. Hide Python to restore focus to previous app
        hide_script = '''
        tell application "System Events"
            set visible of process "Python" to false
        end tell
        '''
        subprocess.run(['osascript', '-e', hide_script])
        
        # 3. Wait for macOS WindowServer to transition
        time.sleep(0.5)
        
        # 4. Click mouse breadcrumb to defeat web app DOM blur
        self.root.after(0, self._execute_mouse_click)
        time.sleep(0.3)
            
        # 5. Wait for user to release Right Option key
        time.sleep(0.5)
        
        # 6. Paste via AppleScript
        paste_script = '''
        tell application "System Events"
            keystroke "v" using command down
        end tell
        '''
        subprocess.run(['osascript', '-e', paste_script])

    # ──────────────────────────────────────────────────────────────────
    # Entry Point
    # ──────────────────────────────────────────────────────────────────
    def run(self):
        print("=" * 50)
        print("🎙️  SiriNo Whisperer — Bulletproof Edition")
        print(f"    Mode:  {MODE.upper()}")
        print(f"    Model: {MODEL}")
        print("=" * 50)
        
        if MODE == "cloud" and not os.environ.get("OPENAI_API_KEY"):
            print("⚠️ WARNING: OPENAI_API_KEY environment variable is not set!")
        
        if MODE == "local":
            print("⏳ Warming up MLX model (first run downloads ~244MB)...")
            try:
                mlx_whisper.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), path_or_hf_repo=MODEL)
                print("✅ Model loaded and ready!")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                
        print(f"🎧 Listening for hotkey... Tap '{HOTKEY}' to toggle recording.")
        print("   Press Ctrl+C in terminal to exit.\n")
        
        # Start Tkinter main loop (blocks here)
        self.root.mainloop()


if __name__ == "__main__":
    app = SiriNoWhispererApp()
    app.run()
