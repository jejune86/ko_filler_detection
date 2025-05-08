import os
import sounddevice as sd
import soundfile as sf
import time

CLASSES = ["filler", "stutter", "fluent"]
SAVE_DIR = "dataset"
SAMPLE_RATE = 16000
DURATION = 3  # 초

def record_sample(class_label, index):
    os.makedirs(os.path.join(SAVE_DIR, class_label), exist_ok=True)
    print(f"🎙️ [{class_label.upper()}] Recording sample {index+1}")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    fname = os.path.join(SAVE_DIR, class_label, f"{class_label}_{index}.wav")
    sf.write(fname, audio, SAMPLE_RATE)
    print(f"✅ Saved: {fname}\n")

def main():
    for label in CLASSES:
        print(f"\n🔹 Start recording for class: {label.upper()}")
        n = int(input(f"How many samples for '{label}'? "))
        for i in range(n):
            input(f"Press Enter to record sample {i+1}...")
            record_sample(label, i)
            time.sleep(0.5)

if __name__ == "__main__":
    main()
