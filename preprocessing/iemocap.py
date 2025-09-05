import pandas as pd
import os
import re
from core.config import CONFIG

class IemocapPreprocessor:
    def __init__(self, dataset_path: str, sessions_count: int = 5):
        self._dataset_path = dataset_path
        self._sessions_count = sessions_count
        # Regex to find lines like "[start_time - end_time] dialog_id emotion [valence, arousal, dominance]"
        self._info_line_pattern = re.compile(r"\[\d+\.\d+ - \d+\.\d+\]\s+([^\s]+)\s+([^\s]+)\s+\[.+\]", re.IGNORECASE)
        # Regex to find lines like "dialog_id [start_time-end_time]: text"
        self._text_line_pattern = re.compile(r"([^\s]+)\s+\[\d+\.\d+-\d+\.\d+\]:\s+(.+)", re.IGNORECASE)


    def generate_dataframe(self) -> pd.DataFrame:
        audios_paths = [] # Changed to store full paths
        audio_filenames = [] # Added to store only filenames
        emotions = []
        texts = []
        sessions = []

        # 从配置中获取目标情感
        # Use training dataset emotions for preprocessing
        target_emotions = CONFIG.dataset_emotions(CONFIG.training_dataset_name())
        print(f"[INFO] Target emotions being extracted: {target_emotions}")

        for session_num in range(1, self._sessions_count + 1):
            session_dir_name = f"Session{session_num}"
            emo_eval_dir = os.path.join(
                self._dataset_path, session_dir_name, "dialog", "EmoEvaluation"
            )
            transcriptions_dir = os.path.join(
                self._dataset_path, session_dir_name, "dialog", "transcriptions"
            )
            session_wav_dir = os.path.join( # Added for constructing full audio path
                self._dataset_path, session_dir_name, "sentences", "wav"
            )


            # 确保目录存在
            if not os.path.isdir(emo_eval_dir):
                print(f" Directory not found, skipping: {emo_eval_dir}")
                continue

            eval_files = [f for f in os.listdir(emo_eval_dir) if f.endswith('.txt')]

            for file in eval_files:
                eval_path = os.path.join(emo_eval_dir, file)
                transcription_path = os.path.join(transcriptions_dir, file)

                if not os.path.exists(transcription_path):
                    continue

                # 1. Read and process transcription file into a dictionary
                transcription_map = {}
                with open(transcription_path, "r") as text_file:
                    for line in text_file:
                        match = self._text_line_pattern.match(line.strip())
                        if match:
                            dialog_id = match.group(1)
                            text_content = match.group(2)
                            transcription_map[dialog_id] = text_content
                        # else:
                            # Optionally, print a warning for lines that don't match the pattern
                            # print(f"[WARNING] Skipping transcription line with unexpected format in {file}: {line.strip()}")


                # 2. Read and process evaluation file, using transcription_map for lookup
                with open(eval_path, "r") as eval_file:
                    for line in eval_file:
                        # Skip lines that don't match the info line pattern (e.g., header)
                        match = self._info_line_pattern.match(line.strip())
                        if not match:
                            continue

                        wav_file_name = match.group(1)
                        emotion = match.group(2)

                        # Use config to filter emotions
                        if emotion not in target_emotions:
                            continue

                        # Look up the corresponding text in the transcription map
                        if wav_file_name in transcription_map:
                            text = transcription_map[wav_file_name]

                            # Construct the full audio file path
                            # IEMOCAP structure: SessionX/sentences/wav/dialog_id_without_last_5_chars/dialog_id.wav
                            try:
                                parent_dir_name = wav_file_name[:-5] # Remove .wav part for parent directory
                                full_audio_path = os.path.join(
                                    session_wav_dir,
                                    parent_dir_name,
                                    f"{wav_file_name}.wav" # Add .wav extension back for the file name
                                )

                                # Check if the audio file actually exists
                                if not os.path.exists(full_audio_path):
                                    print(f"[WARNING] Audio file not found: {full_audio_path}. Skipping.")
                                    continue # Skip this entry if the audio file doesn't exist

                                audios_paths.append(full_audio_path) # Append the full path
                                audio_filenames.append(f"{wav_file_name}.wav") # Append the filename
                                emotions.append(emotion)
                                texts.append(text)
                                sessions.append(session_dir_name)

                            except Exception as e:
                                print(f"[WARNING] Error constructing path for {wav_file_name}: {e}. Skipping.")
                                continue


        # 修正：在所有会话处理完毕后，一次性创建DataFrame
        df = pd.DataFrame({
            "audio_path": audios_paths, # Changed column name
            "audio_filename": audio_filenames, # Added filename column
            "text": texts,
            "emotion": emotions,
            "session": sessions
        })

        print(f"\n[INFO] Preprocessing complete. Total entries extracted: {len(df)}")
        print("[INFO] Emotion distribution:")
        print(df['emotion'].value_counts())
        print("\n[INFO] Data per session:")
        print(df['session'].value_counts())

        return df