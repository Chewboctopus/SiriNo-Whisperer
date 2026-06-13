import os
import time
import tempfile
import threading
import subprocess
import numpy as np
import sounddevice as sd
import soundfile as sf
import mlx_whisper
from pynput import keyboard
import tkinter as tk
import queue
import signal
import sys
import dotenv

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
# If a rogue background process is already running, this will automatically
# hunt it down and kill it so you always get a fresh start!
def kill_previous_instance():
    pid_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sirino.pid")
    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as f:
                old_pid = int(f.read().strip())
            
            # SECURITY FIX: Verify the PID actually belongs to SiriNo Whisperer before murdering it!
            # If macOS recycled the PID and gave it to Chrome, we do NOT want to kill Chrome.
            import subprocess
            cmd = subprocess.run(['ps', '-p', str(old_pid), '-o', 'command='], capture_output=True, text=True).stdout
            if "python" in cmd.lower() and "main.py" in cmd.lower():
                os.kill(old_pid, signal.SIGKILL)
                print(f"💀 Killed old background process (PID {old_pid}) to ensure a fresh start.")
        except Exception:
            pass
    try:
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))
    except:
        pass

kill_previous_instance()

# Configuration
MODE = os.environ.get("WHISPER_MODE", "local").lower()
HOTKEY = 'Right Option (Toggle)'

# mlx-community/whisper-tiny-mlx = Absolute fastest, lowest accuracy
# mlx-community/whisper-base-mlx = Excellent speed, much better accuracy
# mlx-community/whisper-large-v3-turbo = State of the art, but requires more RAM
MODEL = "mlx-community/whisper-base-mlx"
SAMPLE_RATE = 16000
CHUNK_INTERVAL = 0.5  # Transcribe every 0.5 seconds

# Custom Jargon Dictionary
# Automatically loads from your private 'jargon.txt' file so it doesn't get pushed to GitHub!
JARGON = ""
jargon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jargon.txt")
if os.path.exists(jargon_path):
    with open(jargon_path, "r") as f:
        JARGON = f.read().strip()

# Word Replacements (Banned Words & Auto-Correct)
# If the AI constantly mishears a word, or you want to ban filler words like "um", 
# you can force it to replace them here! (To completely ban a word, replace it with "")
REPLACEMENTS = {
    " gonna ": " going to ",
    " gonna.": " going to.",
    " um ": " ",
    " uh ": " ",
    "free pick": "Freepik",
    "free pic": "Freepik"
}

class SiriNoWhispererApp:
    def __init__(self):
        self.recording = False
        self.audio_data = []
        self.audio_lock = threading.Lock()
        self.last_transcribe_time = 0
        self.text_prefix = ""
        self.session_id = 0
        self.target_app = ""
        
        # Setup UI
        self.root = tk.Tk()
        
        # Hide the Python Rocket Icon from the Dock and Cmd-Tab menu
        try:
            from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
            app = NSApplication.sharedApplication()
            app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        except Exception:
            pass
            
        self.root.title("SiriNo Whisperer Preview")
        self.root.attributes('-topmost', True) # Always on top
        self.root.attributes('-alpha', 0.95) # Mostly opaque for readability
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
            insertbackground='white', # Blinking cursor color
            padx=15,
            pady=15,
            borderwidth=0,
            highlightthickness=0
        )
        self.text_widget.pack(expand=True, fill='both')
        
        # Bindings for Enter (Submit) and Escape (Cancel)
        self.text_widget.bind("<Return>", self.on_enter)
        self.text_widget.bind("<Escape>", self.on_escape)
        
        # Prevent the red 'X' button from killing the daemon
        self.root.protocol("WM_DELETE_WINDOW", self.on_close_window)
        
        # Thread-safe UI update queue
        self.ui_queue = queue.Queue()
        self.root.after(50, self.process_ui_queue)
        
        # Hide initially
        self.root.withdraw()
        
        # Audio & Processing
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=self.audio_callback)
        self.stream.start()
        
        # Threads
        self.processing_thread = threading.Thread(target=self.transcription_loop, daemon=True)
        self.processing_thread.start()
        
        self.listener = keyboard.Listener(on_release=self.on_release)
        self.listener.start()

    def process_ui_queue(self):
        try:
            while True:
                func = self.ui_queue.get_nowait()
                func()
        except queue.Empty:
            pass
        self.root.after(50, self.process_ui_queue)

    def audio_callback(self, indata, frames, time_info, status):
        if self.recording:
            with self.audio_lock:
                self.audio_data.append(indata.copy())

    def on_release(self, key):
        # Right Option: Start OR Submit
        if key == keyboard.Key.alt_r:
            if self.root.state() == "withdrawn":
                # Start fresh recording
                self.ui_queue.put(self.start_recording)
            else:
                # If window is open, Right Option submits and pastes globally!
                self.ui_queue.put(self.submit_and_paste)
                
        # Right Command: Pause / Resume (Only works if a session is active)
        elif key == keyboard.Key.cmd_r:
            if self.root.state() != "withdrawn":
                if self.recording:
                    self.ui_queue.put(self.stop_recording)
                else:
                    self.ui_queue.put(self.start_recording)

    def play_sound(self, sound_name):
        # SECURITY FIX: Use native macOS AppKit to play sounds in-memory.
        # This prevents spawning hundreds of `afplay` zombie subprocesses!
        try:
            from AppKit import NSSound
            sound = NSSound.soundNamed_(sound_name)
            if sound:
                sound.play()
        except Exception:
            pass

    def start_recording(self):
        print("🎤 Recording started...")
        self.play_sound("Ping")
        self.recording = True
        self.root.title("SiriNo Whisperer - 🔴 LIVE (Tap Right Option to Pause)")
        
        # If the window is already visible, the user is resuming a paused recording.
        # We must save their edited text before clearing the audio buffer!
        if self.root.state() != "withdrawn":
            current_text = self.text_widget.get("1.0", "end-1c").strip()
            if current_text and current_text != "Listening...":
                self.text_prefix = current_text + " "
            else:
                self.text_prefix = ""
        else:
            self.session_id += 1
            self.text_prefix = ""
            
            # Remember the exact app the user was in before we steal focus!
            try:
                from AppKit import NSWorkspace
                self.target_app = NSWorkspace.sharedWorkspace().frontmostApplication().localizedName()
            except Exception:
                self.target_app = ""
                
            self.text_widget.delete("1.0", tk.END)
            self.text_widget.insert(tk.END, "Listening...")
            self.root.deiconify() # Show window
            self.root.lift()      # Force to top
            self.root.update()    # Force render
            
            # Force the CURRENT Python process to take keyboard focus (System Events prevents spawning duplicate ghost apps)
            try:
                subprocess.Popen(['osascript', '-e', 'tell application "System Events" to set frontmost of process "Python" to true'])
            except:
                pass
            self.root.focus_force()
            self.text_widget.focus_set()
            
        # WIPE the audio buffer! If we don't do this, resuming will cause Whisper to
        # re-transcribe the old audio and duplicate the text!
        with self.audio_lock:
            self.audio_data = [] # Reset audio buffer for the new words

    def stop_recording(self):
        print("⏸️ Recording paused (Editable).")
        self.play_sound("Pop")
        self.recording = False
        self.root.title("SiriNo Whisperer - ⏸️ PAUSED (Edit your text, then press Enter to Paste)")
        # Window stays open so the user can edit!

    def on_enter(self, event):
        self.submit_and_paste()
        return "break" # Prevents entering a newline

    def on_escape(self, event):
        self.recording = False
        self.root.withdraw()
        return "break"

    def on_close_window(self):
        self.recording = False
        self.root.withdraw()

    def submit_and_paste(self):
        self.recording = False
        
        # Extract the final edited text directly from the UI widget
        final_text = self.text_widget.get("1.0", "end-1c").strip()
        if final_text == "Listening...":
            final_text = ""
            
        self.root.withdraw() # Hide window
        
        if final_text:
            # Run paste in background thread to prevent AppleScript delay from blocking Tkinter
            threading.Thread(target=self.paste_text_mac, args=(final_text,), daemon=True).start()

    def transcription_loop(self):
        while True:
            if self.recording and MODE == "local":
                now = time.time()
                if now - self.last_transcribe_time > CHUNK_INTERVAL:
                    with self.audio_lock:
                        if len(self.audio_data) > 0:
                            self.last_transcribe_time = now
                            # Copy buffer safely
                            audio_copy = self.audio_data.copy()
                        else:
                            audio_copy = []
                            
                    if len(audio_copy) > 0:
                        audio_np = np.concatenate(audio_copy, axis=0)
                        
                        if len(audio_np) > SAMPLE_RATE * 0.2: # At least 0.2s of audio
                            # --- VAD Gate (Voice Activity Detection) ---
                            # A completely silent Mac mic sits around 0.002. Speech hits 0.1+.
                            # If the volume is under 0.015, they aren't speaking!
                            if np.max(np.abs(audio_np)) < 0.015:
                                # If they've been silent for 5 seconds, wipe the buffer to save RAM
                                if len(audio_np) > SAMPLE_RATE * 5:
                                    with self.audio_lock:
                                        self.audio_data = []
                                continue # Skip AI transcription completely!
                                
                            text = self.transcribe_chunk(audio_np, self.session_id)
                            
                            # Auto-Commit: If the buffer is over 15 seconds, bake the text into the prefix permanently!
                            if len(audio_np) > SAMPLE_RATE * 15 and text:
                                self.text_prefix += text + " "
                                with self.audio_lock:
                                    self.audio_data = [] # WIPE buffer
                                # Render the UI with an empty live-preview because we just committed it!
                                self.ui_queue.put(lambda sid=self.session_id: self.update_text_widget("", sid))
                            else:
                                # Normal live preview render
                                self.ui_queue.put(lambda t=text, sid=self.session_id: self.update_text_widget(t, sid))
                                
            time.sleep(0.1)

    def is_hallucination(self, text):
        lower_text = text.lower()
        if "thank you for watching" in lower_text or "thanks for watching" in lower_text:
            return True
        if text.strip() == "You" or text.strip() == "I":
            return True
        import zlib
        encoded = text.encode('utf-8')
        if len(encoded) > 30 and (len(zlib.compress(encoded)) / len(encoded)) < 0.35:
            print("🧹 Suppressed repetitive Whisper hallucination!")
            return True
        return False

    def transcribe_chunk(self, audio_np, session_id):
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)
        try:
            sf.write(temp_path, audio_np, SAMPLE_RATE)
            
            # Use Whisper on the local Mac GPU
            # condition_on_previous_text=False is CRITICAL for live streaming.
            # Otherwise, if it hears loud TV noise, it tries to guess what you said
            # by just regurgitating the last sentence it successfully transcribed!
            result = mlx_whisper.transcribe(
                temp_path, 
                path_or_hf_repo=MODEL, 
                condition_on_previous_text=False,
                initial_prompt=JARGON
            )
            text = result.get('text', '').strip()
            if text:
                if self.is_hallucination(text):
                    return ""
                
                # Apply custom word replacements and banned words
                for bad_word, good_word in REPLACEMENTS.items():
                    # We do case-insensitive replacement to catch it anywhere in the sentence
                    import re
                    text = re.sub(re.escape(bad_word), good_word, text, flags=re.IGNORECASE)
                
                # Fix capitalizations at the start of the string if it was replaced
                if len(text) > 0:
                    text = text[0].upper() + text[1:]
                
                return text
            
            return ""
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def update_text_widget(self, text, session_id):
        # Drop stale updates if the user has already submitted and started a new recording
        if self.session_id != session_id:
            return
            
        # We replace the text with the live streaming results, prefixed with any previously saved edits.
        # Note: If the user is actively typing, this will overwrite their typing.
        full_text = self.text_prefix + text
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert(tk.END, full_text)
        self.text_widget.see(tk.END)
        
        # Real-time crash backup
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backup.txt"), "w") as f:
                f.write(full_text)
        except:
            pass

    def paste_text_mac(self, text):
        escaped_text = text.replace('\\', '\\\\').replace('"', '\\"')
        
        # Explicitly reactivate the app the user was in when they started recording
        activate_script = "delay 0.2"
        if getattr(self, 'target_app', ""):
            # SECURITY FIX: Intelligent polling loop. Wait exactly until the target app 
            # confirms it has stolen keyboard focus back before we fire the Paste command!
            activate_script = f'''
            tell application "{self.target_app}" to activate
            repeat 20 times
                tell application "System Events"
                    try
                        if frontmost of application process "{self.target_app}" then exit repeat
                    end try
                end tell
                delay 0.05
            end repeat
            '''
            
        script = f'''
        {activate_script}
        tell application "System Events"
            set the clipboard to "{escaped_text}"
            keystroke "v" using command down
        end tell
        '''
        subprocess.run(['osascript', '-e', script])

    def run(self):
        print("="*40)
        print("🎙️ SiriNo Whisperer Background Service (Live Editable UI)")
        print(f"Mode: {MODE.upper()}")
        
        if MODE == "cloud" and not os.environ.get("OPENAI_API_KEY"):
            print("⚠️ WARNING: OPENAI_API_KEY environment variable is not set!")
        
        if MODE == "local":
            print("Loading local model...")
            try:
                mlx_whisper.transcribe(np.zeros(16000, dtype=np.float32), path_or_hf_repo=MODEL)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {e}")
                
        print(f"Listening for hotkey... Tap '{HOTKEY}' to toggle recording.")
        print("Press Ctrl+C in terminal to exit.")
        
        # Start Tkinter main loop
        self.root.mainloop()

if __name__ == "__main__":
    app = SiriNoWhispererApp()
    app.run()
