from openai import OpenAI
# for backward compatibility, you can still use `https://api.deepseek.com/v1` as `base_url`.
client = OpenAI(api_key="sk-47799b5e6d594b6d9edf6890dd193224", base_url="https://api.deepseek.com")
print(client.models.list())