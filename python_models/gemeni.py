from google import genai

client = genai.Client(api_key="AIzaSyBxnTU1TriRU9HJEXqnLpb-gwTvka8g_SU")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Explain how AI works",
)

print(response.text)