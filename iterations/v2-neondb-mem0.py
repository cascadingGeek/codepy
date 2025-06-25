import asyncio
from dotenv import load_dotenv
import os
from openai import OpenAI
from agents import Agent, WebSearchTool, Runner
from mem0 import Memory

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")

if not url:
    raise Exception("SUPABASE_URL not found. Please set the SUPABASE_URL environment variable.")

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": os.getenv("MODEL", "gpt-4o-mini"),
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
    },
    # "vectorstore": {
    #     "provider": "supabase",
    #     "config": {
    #         "connection_string": url,
    #         "collection_name": "memories",
    #     },
    # }
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

sport_agent = Agent(
    name="Sport Agent",
    instructions=(
        "You are a helpful AI sport agent. If asked about questions related to sports, use the web search tool."
        ),
    model="gpt-4o-mini",
    tools=[WebSearchTool()],
)

async def chat_with_memories(message: str, user_id: str = "default-user") -> str:
    relevant_memories = memory.search(query=message, user_id=user_id, limit=3)
    memory_results = relevant_memories["results"]

    if memory_results:
        memories_str = "\n".join(f"- {entry['memory']}" for entry in memory_results)
        system_prompt = f"You are a sports assistant. Use the user's memories to respond.\nUser Memories:\n{memories_str}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        assistant_response = response.choices[0].message.content
    else:
        print("🌐 No memory match. Using web search...")
        assistant_response = await Runner.run(sport_agent, message)

    memory.add(
        [{"role": "user", "content": message},
         {"role": "assistant", "content": assistant_response}],
        user_id=user_id
    )

    return assistant_response

async def main():
    print("⚽ Chat with Sport Agent (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break
        response = await chat_with_memories(user_input)
        print(f"Sport Agent: {response}")

if __name__ == "__main__":
    asyncio.run(main())
