from google import genai
from google.genai import errors

client = genai.Client()

question = input("Ask me a question: ")

try:
    interaction = client.interactions.create(
        model="gemini-3.5-flash-lite",
        input=question
    )

    print(interaction.output_text)

except errors.APIError as e:
    print("Gemini API error!")
    print("Status code:", e.code)
    print("Message:", e.message)

except Exception as e:
    print("Unexpected error:")
    print(e)