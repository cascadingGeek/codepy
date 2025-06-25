from dotenv import load_dotenv
from mem0 import Memory
import os

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

if not url:
    raise Exception("SUPABASE_URL not found. Please set the SUPABASE_URL environment variable.")

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": os.getenv("MODEL", "gpt-4o-mini"),
        },
    },
    # "vectorstore": {
#         "provider": "supabase",
#         "config": {
#             "connection_string": url,
#             "collection_name": "memories",
#         },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "path": "database/qdrant_storage",
            "collection_name": "memories",
            "on_disk": True,  
        }
    },
    "history_db_path": "database/history.db",  
    "version": "v1.1"

}

openai_client = OpenAI()
memory = Memory.from_config(config)

def chat_with_memory(message: str, user_id: str = "default-user") -> str:
    relevant_memories = memory.search(query=message, user_id=user_id, limit=3)
    memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories["results"])
    
    print(f"Relevant Memories: {relevant_memories}\n")

    system_prompt = f"You are a helpful AI. Answer the user's question based on query and memories.\nUser Memories:\n{memories_str}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]
    
    response = openai_client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    assistant_response = response.choices[0].message.content

    messages.append({"role": "assistant", "content": assistant_response})
    result = memory.add(messages, user_id=user_id)
    
    print(f"Memory added: {result}\n")

    return assistant_response

def main():
    print("Chat with AI (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = chat_with_memory(user_input)
        print(f"AI: {response}")

if __name__ == "__main__":
    main()
