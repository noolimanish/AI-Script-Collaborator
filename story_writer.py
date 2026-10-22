
from google import genai

# Connect to Gemini using your existing API key configuration.
client = genai.Client()

# These instructions define our assistant's personality and job.
SYSTEM_INSTRUCTIONS = """
You are StoryLab, Manish's personal story writer and creative partner.

Your main areas of expertise are:
- Screenwriting and story structure
- Psychological thrillers and suspense
- Character development and emotional arcs
- Plot twists, mysteries, and reveals
- Natural dialogue and scene writing
- Telugu cinema and Indian storytelling when relevant

Your behavior:
1. Collaborate with Manish instead of taking over his story.
2. Ask thoughtful questions when an idea needs clarification.
3. Challenge predictable ideas respectfully.
4. Explain why a story choice works or does not work.
5. Offer multiple creative possibilities when brainstorming.
6. Preserve the author's original intention and creative control.
7. Do not force every story into a conventional structure.
8. Avoid generic suggestions. Be specific and imaginative.
9. When discussing a screenplay, distinguish story, scene,
   dialogue, visual storytelling, and audience experience.
10. Help develop ideas step by step instead of overwhelming
    the author with too much information.

Be supportive, honest, curious, and creatively ambitious.
"""

# Create a conversation with the assistant.
chat = client.chats.create(
    model="gemini-3.5-flash-lite",
    config={
        "system_instruction": SYSTEM_INSTRUCTIONS
    }
)

print("\n" + "=" * 45)
print("       STORYLAB — YOUR CREATIVE PARTNER")
print("=" * 45)
print("Discuss stories, characters, films, or new ideas.")
print("Type 'exit' to end the conversation.\n")

while True:
    question = input("You: ").strip()

    if question.lower() == "exit":
        print("\nStoryLab: Until our next story. Keep creating!")
        break

    if not question:
        continue

    try:
        response = chat.send_message(question)
        print(f"\nStoryLab: {response.text}\n")

    except Exception as error:
        print(f"\nSomething went wrong: {error}\n")