import pandas as pd
import os
import re
from core.config import CONFIG

class CremaDPreprocessor:
    def __init__(self, dataset_path: str):
        self._dataset_path = dataset_path
        # Regex to extract info from CREMA-D filename:
        # ActorID_Emotion_EmotionLevel_Statement.wav
        # Example: 1001_HAP_SAD_XX.wav -> ActorID=1001, Emotion=HAP, EmotionLevel=SAD, Statement=XX
        # Note: The actual emotion is the second part (e.g., HAP)
        self._filename_pattern = re.compile(r"(\d+)_([A-Z]+)_([A-Z]+)_([A-Z]+)\.wav")

        # 新增：存储文本缩写与完整句子的映射关系
        self._text_map = {
            "IEO": "It's eleven o'clock",
            "TIE": "That is exactly what happened",
            "IOM": "I'm on my way to the meeting",
            "IWW": "I wonder what this is about",
            "TAI": "The airplane is almost full",
            "MTI": "Maybe tomorrow it will be cold",
            "IWL": "I would like a new alarm clock",
            "ITH": "I think I have a doctor's appointment",
            "DFA": "Don't forget a jacket",
            "ITS": "I think I've seen this before",
            "TSI": "The surface is slick",
            "WSI": "We'll stop in a couple of minutes"
        }

    def generate_dataframe(self) -> pd.DataFrame:
        audios_paths = [] # Changed to store full paths
        audio_filenames = [] # Added to store only filenames
        emotions = []
        texts = [] # CREMA-D file names don't contain text, will use a placeholder or common text if available
        speakers = [] # Speaker ID from filename

        # Get target emotions for CREMA-D from config
        target_emotions = CONFIG.dataset_emotions(CONFIG.evaluation_dataset_name())
        print(f"[INFO] Target emotions being extracted for CREMA-D: {target_emotions}")

        # CREMA-D structure: AudioWAV/ directories
        audio_dir = os.path.join(self._dataset_path, "AudioWAV")

        if not os.path.isdir(audio_dir):
             print(f"[ERROR] CREMA-D Audio directory not found: {audio_dir}")
             # Changed to return DataFrame with the new column name
             return pd.DataFrame(columns=["audio_path", "audio_filename", "text", "emotion", "speaker"]) # Return empty DataFrame

        wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

        for file in wav_files:
            match = self._filename_pattern.match(file)
            if not match:
                # print(f"[WARNING] Skipping file with unexpected format: {file}")
                continue

            speaker_id = match.group(1)
            text_abbr = match.group(2)    # The second part is the sentence acronym, e.g., 'IEO'
            # CREMA-D emotion labels: SAD, NEU, ANG, DIS, FEA, HAP
            emotion_label = match.group(3) # The third part is the primary emotion

            # 使用映射字典将缩写转换为完整句子
            # .get() 方法更安全, 如果找不到键, 会返回默认值 (这里是空字符串)
            text_content = self._text_map.get(text_abbr, "")

            # Map CREMA-D labels to the desired emotion labels if necessary
            # Assuming the target emotions in config match the CREMA-D labels (e.g., ANG for anger)
            # If mapping is needed, add a dictionary here, e.g., crema_map = {'ANG': 'ang', ...}
            # For now, assuming direct match or subset.
            mapped_emotion = emotion_label.lower() # Convert to lowercase to match config

            # Use config to filter emotions
            if mapped_emotion not in target_emotions:
                continue

            # Construct the full audio file path
            full_audio_path = os.path.join(audio_dir, file)

            # Check if the audio file actually exists
            if not os.path.exists(full_audio_path):
                 print(f"[WARNING] Audio file not found: {full_audio_path}. Skipping.")
                 continue # Skip this entry if the audio file doesn't exist


            audios_paths.append(full_audio_path) # Append the full path
            audio_filenames.append(file) # Append the filename
            emotions.append(mapped_emotion)
            texts.append(text_content)
            speakers.append(speaker_id)


        df = pd.DataFrame({
            "audio_path": audios_paths, # Changed column name
            "audio_filename": audio_filenames, # Added filename column
            "text": texts,
            "emotion": emotions,
            "speaker": speakers
        })

        print(f"\n[INFO] CREMA-D Preprocessing complete. Total entries extracted: {len(df)}")
        print("[INFO] Emotion distribution for CREMA-D:")
        print(df['emotion'].value_counts())
        # Speaker distribution might also be useful
        print("\n[INFO] Data per Speaker:")
        print(df['speaker'].value_counts())


        return df