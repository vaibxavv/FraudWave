import asyncio
import os
import edge_tts

# Neural synthetic voices (Microsoft Azure Neural TTS)
VOICES = [
    ("en-US-GuyNeural", "This is an urgent security verification notice from the central account department."),
    ("en-US-AriaNeural", "Please confirm your six digit one time password to prevent unauthorized account access."),
    ("en-US-ChristopherNeural", "We have detected unusual activity on your card. Verify your credentials immediately."),
    ("en-GB-SoniaNeural", "Hello, I am calling regarding the transfer authorization code requested earlier today.")
]

async def generate():
    os.makedirs("dataset/fake", exist_ok=True)
    print("Generating authentic neural AI voices...")
    
    for idx, (voice, text) in enumerate(VOICES, start=1):
        output_file = f"dataset/fake/fake_{idx}.wav"
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)
        print(f"Generated: {output_file} (Voice: {voice})")

    print("\nAll synthetic voice files generated successfully in dataset/fake/!")

if __name__ == "__main__":
    asyncio.run(generate())