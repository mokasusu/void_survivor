import os
import json

transcript_path = "/Users/vducc3110/.gemini/antigravity-ide/brain/83e38a68-08ef-4c2f-b33c-207cc0581e57/.system_generated/logs/transcript.jsonl"
if not os.path.exists(transcript_path):
    print("Transcript not found at:", transcript_path)
    exit()

print("Searching transcript for all user inputs with image uploads...")
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            content = data.get("content", "")
            step_idx = data.get("step_index", 0)
            source = data.get("source", "")
            step_type = data.get("type", "")
            
            if source == "USER_EXPLICIT" and step_type == "USER_INPUT":
                # Check if this input has images in metadata
                tc_str = json.dumps(data)
                if "media__" in tc_str or "tempmediaStorage" in tc_str:
                    print(f"\n=== USER INPUT STEP {step_idx} ===")
                    print("Text:", content)
                    # Print metadata details if any
                    metadata = data.get("metadata", {})
                    print("Metadata Keys:", list(metadata.keys()) if isinstance(metadata, dict) else "Not dict")
                    # Look for media__ filenames in tc_str
                    words = tc_str.split()
                    media_words = [w for w in words if "media__" in w]
                    print("Associated Media:", set(media_words))
        except Exception as e:
            pass
