import os
import json

transcript_path = "/Users/vducc3110/.gemini/antigravity-ide/brain/83e38a68-08ef-4c2f-b33c-207cc0581e57/.system_generated/logs/transcript.jsonl"
if not os.path.exists(transcript_path):
    print("Transcript not found at:", transcript_path)
    exit()

media_targets = [
    "media__1781026398177.png",
    "media__1781049559730.png",
    "media__1781050399085.png",
    "media__1781050774049.png",
    "media__1781051569991.jpg",
    "media__1781057874240.jpg"
]

print("Searching transcript for target media files...")
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            content = data.get("content", "")
            step_idx = data.get("step_index", 0)
            source = data.get("source", "")
            step_type = data.get("type", "")
            
            found = [m for m in media_targets if m in content or (data.get("tool_calls") and any(m in json.dumps(tc) for tc in data.get("tool_calls")))]
            if found:
                print(f"--- Step {step_idx} (Source: {source}, Type: {step_type}) matches {found} ---")
                if content:
                    print("Content preview:")
                    print(content[:500])
                if "tool_calls" in data and data["tool_calls"]:
                    for tc in data["tool_calls"]:
                        tc_str = json.dumps(tc)
                        if any(m in tc_str for m in media_targets):
                            print("Tool Call:", tc.get("name"), "args:", tc_str[:300])
                print("-" * 50)
        except Exception as e:
            pass
