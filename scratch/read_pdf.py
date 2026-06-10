from pypdf import PdfReader

reader = PdfReader("/Users/vducc3110/Desktop/void_survivor/Trí tuệ nhân tạo_9,5.pdf")
print(f"Number of pages: {len(reader.pages)}")

text_content = []
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    text_content.append(f"--- PAGE {i+1} ---")
    text_content.append(text)

with open("/Users/vducc3110/Desktop/void_survivor/scratch/pdf_text.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(text_content))

print("Text extracted and saved to scratch/pdf_text.txt")
