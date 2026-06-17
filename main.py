import os
import re
import gc
import sys
import time
import zlib
import queue
import signal
import threading
import subprocess

import numpy as np
import sounddevice as sd
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
DEBUG = os.environ.get("WHISPER_DEBUG", "").strip() == "1"
HOTKEY = 'Right Option (Toggle)'

# mlx-community/whisper-tiny-mlx        = Absolute fastest, lowest accuracy
# mlx-community/whisper-base-mlx        = Excellent speed, much better accuracy
# mlx-community/whisper-small-mlx       = Balanced: good accuracy, fast (~0.5s per call)
# mlx-community/whisper-large-v3-turbo  = State of the art, but SLOW (~3s per call, causes UI lag)
MODEL = os.environ.get("WHISPER_MODEL", "mlx-community/whisper-small-mlx")
SAMPLE_RATE = 16000
CHUNK_INTERVAL = 0.5  # Minimum seconds between dispatch checks

# Tuning constants
MAX_TRANSCRIBE_SECONDS = 10   # Never send more than this to MLX (prevents GPU stalls)
AUTO_COMMIT_SECONDS = 8       # Bake text into prefix before buffer hits the cap
AUTO_COMMIT_OVERLAP = 1       # Seconds of audio to keep after auto-commit for acoustic context
VAD_SILENCE_THRESHOLD = 0.015 # Amplitude below this = silence (Mac mic noise floor ~0.002)
VAD_TAIL_SECONDS = 0.5        # Only check the most recent audio for silence (not the whole buffer)
VAD_WIPE_SECONDS = 5          # Wipe buffer after this much continuous silence
BACKUP_INTERVAL = 5           # Only write crash backup every N seconds (prevents I/O spam)
MAX_TRANSCRIBE_SAMPLES = SAMPLE_RATE * MAX_TRANSCRIBE_SECONDS
AUTO_COMMIT_SAMPLES = SAMPLE_RATE * AUTO_COMMIT_SECONDS
AUTO_COMMIT_OVERLAP_SAMPLES = SAMPLE_RATE * AUTO_COMMIT_OVERLAP
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
        self.last_inference_duration = 0.5  # Adaptive dispatch cadence seed
        self.last_backup_time = 0

        # --- Session Transcript Log ---
        self.session_log = []

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
        
        # --- Audio (created but NOT started — mic activates on-demand) ---
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, callback=self._audio_callback
        )
        
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
        
        # Activate the microphone
        if not self.stream.active:
            self.stream.start()
        
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
        self.stream.stop()  # Release the mic
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
        self.stream.stop()  # Release the mic
        with self.audio_lock:
            self.text_prefix = ""
            self.text_suffix = ""
            self.audio_data = []
        self.root.withdraw()

    def submit_and_paste(self):
        self.recording = False
        self.is_window_open = False
        self.stream.stop()  # Release the mic
        
        final_text = self.text_widget.get("1.0", "end-1c").strip()
        self.text_widget.delete("1.0", tk.END)
        self.root.withdraw()
        
        with self.audio_lock:
            self.text_prefix = ""
            self.text_suffix = ""
            self.audio_data = []
        
        if final_text and final_text != "Listening...":
            # Log to session transcript
            entry = {
                "index": len(self.session_log) + 1,
                "time": time.strftime("%I:%M:%S %p"),
                "text": final_text
            }
            self.session_log.append(entry)
            print(f"\n{'─' * 50}")
            print(f"📝 SESSION TRANSCRIPT ({len(self.session_log)} entries)")
            print(f"{'─' * 50}")
            for e in self.session_log:
                print(f"  {e['index']:>3}. [{e['time']}] {e['text']}")
            print(f"{'─' * 50}\n")
            threading.Thread(target=self._paste_text_mac, args=(final_text,), daemon=True).start()

    # ──────────────────────────────────────────────────────────────────
    # Dispatch Loop (background thread — prepares audio, never blocks)
    # ──────────────────────────────────────────────────────────────────
    def _dispatch_loop(self):
        """Checks for new audio and dispatches work to the persistent worker thread.
        If the worker is still busy, the work item is silently dropped (queue maxsize=1).
        This guarantees the loop NEVER blocks regardless of how slow MLX is.
        
        Uses adaptive cadence: dispatch interval scales with actual inference time
        to prevent queue pileup when the model is slow."""
        while True:
            try:
                if self.recording and MODE == "local":
                    now = time.time()
                    # Adaptive cadence: wait at least 1.2x the last inference duration
                    adaptive_interval = max(CHUNK_INTERVAL, self.last_inference_duration * 1.2)
                    if now - self.last_transcribe_time > adaptive_interval:
                        self.last_transcribe_time = now
                        
                        # Lightweight VAD peek — only check tail chunk, no full concatenation
                        with self.audio_lock:
                            if len(self.audio_data) == 0:
                                time.sleep(0.1)
                                continue
                            # Count total samples without concatenating
                            total_samples = sum(chunk.shape[0] for chunk in self.audio_data)
                        
                        if total_samples < MIN_AUDIO_SAMPLES:
                            time.sleep(0.1)
                            continue
                            
                        # --- VAD Gate (tail-only — peek at last chunk, no full copy) ---
                        with self.audio_lock:
                            tail_chunk = self.audio_data[-1]
                        tail_is_silent = np.max(np.abs(tail_chunk)) < VAD_SILENCE_THRESHOLD
                        
                        if tail_is_silent:
                            # For wipe check, scan all chunks without concatenating
                            if total_samples > VAD_WIPE_SAMPLES:
                                with self.audio_lock:
                                    all_silent = all(np.max(np.abs(c)) < VAD_SILENCE_THRESHOLD for c in self.audio_data)
                                if all_silent:
                                    with self.audio_lock:
                                        self.audio_data = []
                                    if DEBUG:
                                        print(f"🧹 [{time.strftime('%H:%M:%S')}] VAD wiped {total_samples/SAMPLE_RATE:.1f}s of silent buffer")
                            time.sleep(0.1)
                            continue
                        
                        if DEBUG:
                            print(f"📡 [{time.strftime('%H:%M:%S')}] Dispatch: buffer={total_samples/SAMPLE_RATE:.1f}s, "
                                  f"tail_amp={np.max(np.abs(tail_chunk)):.4f}, cadence={adaptive_interval:.2f}s, "
                                  f"queue={'full' if self.work_queue.full() else 'ready'}")
                        
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
                # Block until signaled, with timeout for immediate re-check
                # after completing a transcription (eliminates dead time gap)
                try:
                    session_id = self.work_queue.get(timeout=0.3)
                except queue.Empty:
                    # No dispatch signal, but if we're actively recording, self-trigger
                    if self.recording and MODE == "local":
                        session_id = self.session_id
                    else:
                        continue
                
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
                
                # Strip echoed words from the overlap audio
                with self.audio_lock:
                    text = self._strip_overlap_echo(self.text_prefix, text)
                
                # --- Auto-Commit ---
                if buffer_length > AUTO_COMMIT_SAMPLES and text:
                    with self.audio_lock:
                        self.text_prefix += text + " "
                        # Keep last 1s of audio for acoustic context (prevents word fragmentation)
                        if len(audio_np) > AUTO_COMMIT_OVERLAP_SAMPLES:
                            self.audio_data = [audio_np[-AUTO_COMMIT_OVERLAP_SAMPLES:]]
                        else:
                            self.audio_data = []
                    # Drain any stale signals that reference pre-wipe state
                    try:
                        self.work_queue.get_nowait()
                    except queue.Empty:
                        pass
                    if DEBUG:
                        print(f"📌 [{time.strftime('%H:%M:%S')}] Auto-commit: baked {len(text)} chars into prefix, "
                              f"kept {AUTO_COMMIT_OVERLAP}s overlap")
                    self.ui_queue.put(lambda sid=session_id: self._update_text_widget("", sid))
                    # Deferred GC: schedule for 2s later so it doesn't block the hot path
                    threading.Timer(2.0, gc.collect).start()
                else:
                    self.ui_queue.put(lambda t=text, sid=session_id: self._update_text_widget(t, sid))
                
                # Cooldown: prevent back-to-back GPU saturation that starves the UI thread
                time.sleep(0.15)
                    
            except Exception as e:
                print(f"⚠️ Transcription worker error (recovering): {e}")

    # ──────────────────────────────────────────────────────────────────
    # Overlap Deduplication
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _strip_overlap_echo(prefix, new_text):
        """Remove echoed words from the start of new_text that duplicate the
        tail of prefix.  This happens because auto-commit keeps 1s of audio
        overlap for acoustic context, and Whisper re-transcribes those words.

        Uses word-level suffix/prefix matching (case-insensitive) to find the
        longest overlap and strip it from new_text."""
        if not prefix or not new_text:
            return new_text

        prefix_words = prefix.rstrip().lower().split()
        new_words_lower = new_text.lower().split()
        new_words_orig = new_text.split()

        # Check up to 8 words of overlap (1s of audio ≈ 2-4 spoken words,
        # but Whisper may expand context slightly)
        max_check = min(8, len(prefix_words), len(new_words_lower))

        best_overlap = 0
        for overlap_len in range(1, max_check + 1):
            # Does the tail of prefix match the head of new_text?
            if prefix_words[-overlap_len:] == new_words_lower[:overlap_len]:
                best_overlap = overlap_len

        if best_overlap > 0:
            stripped = " ".join(new_words_orig[best_overlap:])
            if DEBUG:
                echo = " ".join(new_words_orig[:best_overlap])
                print(f"🔇 [{time.strftime('%H:%M:%S')}] Stripped {best_overlap}-word overlap echo: '{echo}'")
            return stripped

        return new_text

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
        try:
            # Pass numpy array directly to mlx_whisper — no temp file I/O needed
            audio_flat = audio_np.squeeze().astype(np.float32)
            
            t0 = time.time()
            if DEBUG:
                print(f"🔄 [{time.strftime('%H:%M:%S')}] Transcribing {len(audio_flat)/SAMPLE_RATE:.1f}s of audio...")
            
            result = mlx_whisper.transcribe(
                audio_flat, 
                path_or_hf_repo=MODEL, 
                condition_on_previous_text=False,
                initial_prompt=JARGON
            )
            text = result.get('text', '').strip()
            
            elapsed = time.time() - t0
            # Feed back to adaptive dispatch cadence
            self.last_inference_duration = elapsed
            
            if DEBUG:
                print(f"✅ [{time.strftime('%H:%M:%S')}] Result ({elapsed:.1f}s): '{text[:80]}{'...' if len(text) > 80 else ''}'")
            
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
        
        # Crash backup — throttled to every BACKUP_INTERVAL seconds to prevent I/O spam
        now = time.time()
        if now - self.last_backup_time > BACKUP_INTERVAL:
            self.last_backup_time = now
            def _write_backup(content):
                try:
                    backup_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backup.txt")
                    with open(backup_path, "w") as f:
                        f.write(content)
                except Exception:
                    pass
            threading.Thread(target=_write_backup, args=(full_text,), daemon=True).start()

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
