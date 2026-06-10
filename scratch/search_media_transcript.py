import os
import json

transcript_path = "/Users/vducc3110/.gemini/antigravity-ide/brain/83e38a68-08ef-4c2f-b33c-207cc0581e57/.system_generated/logs/transcript.jsonl"
if not os.path.exists(transcript_path):
    print("Transcript not found at:", transcript_path)
    exit()

media_to_find = [
    "media__1781024159281.jpg",
    "media__1781024824439.jpg",
    "media__1781025045736.png",
    "media__1781025398359.png",
    "media__1781026398177.png",
    "media__1781049559730.png",
    "media__1781050399085.png",
    "media__1781050774049.png",
    "media__1781051569991.jpg",
    "media__1781057874240.jpg"
]

print("Searching transcript for specific media uploads...")
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            content = data.get("content", "")
            step_idx = data.get("step_index", 0)
            source = data.get("source", "")
            step_type = data.get("type", "")
            
            # Check if any target media file is mentioned in user input or model output
            mentioned = [m for m in media_to_find if m in content or os.path.basename(m) in content]
            if mentioned:
                print(f"\n================ STEP {step_idx} (Source: {source}, Type: {step_type}) ================")
                print(content[:1500]) # Print up to 1500 chars of context
                print("========================================================================\n")
        except Exception as e:
            pass
