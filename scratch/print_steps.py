import os
import json

transcript_path = "/Users/vducc3110/.gemini/antigravity-ide/brain/83e38a68-08ef-4c2f-b33c-207cc0581e57/.system_generated/logs/transcript.jsonl"
if not os.path.exists(transcript_path):
    print("Transcript not found at:", transcript_path)
    exit()

print("Printing conversation steps 140 to 165...")
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            step_idx = data.get("step_index", 0)
            source = data.get("source", "")
            step_type = data.get("type", "")
            
            if 140 <= step_idx <= 165:
                print(f"--- STEP {step_idx} (Source: {source}, Type: {step_type}) ---")
                if "content" in data and data["content"]:
                    print(data["content"][:800])
                if "tool_calls" in data and data["tool_calls"]:
                    for tc in data["tool_calls"]:
                        print("  Tool Call:", tc.get("name"), "args:", json.dumps(tc.get("arguments"))[:200])
                print("-" * 50)
        except Exception as e:
            pass
