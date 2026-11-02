import re

with open("ui_src/src/pages/IngestionProfilesPage.tsx", encoding="utf-8") as f:
    content = f.read()

# Find ALL JSX display text content
lines = content.split("\n")
print("=== JSX display text (between >...<) ===")
for i, line in enumerate(lines):
    # Match >TEXT< patterns in JSX
    for m in re.finditer(r">([A-Za-z][A-Za-z0-9 /,.:;\-()]+)<", line):
        text = m.group(1).strip()
        if len(text) > 2 and not text.startswith("div") and "className" not in text:
            print(f"  L{i+1}: {repr(text)}")

print("\n=== title= or placeholder= attributes ===")
for i, line in enumerate(lines):
    for m in re.finditer(r'(title|placeholder)="([A-Za-z][A-Za-z0-9 /,.:;\-()]+)"', line):
        print(f"  L{i+1}: {m.group(1)}={repr(m.group(2))}")
