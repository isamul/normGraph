import asyncio
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from dotenv import load_dotenv
load_dotenv()

from langfuse.callback import CallbackHandler
langfuse_handler = CallbackHandler(
    public_key="pk-lf-a94df179-5c70-4c79-8ddb-719b8c31e30b",
    secret_key="sk-lf-23574838-ebc7-40ae-a3fd-82e443e0cdc3",
    host="https://cloud.langfuse.com"
)

from base_agent.agent import graph

config = {
    "configurable": {"thread_id": "2"},
    "callbacks": [BaseCallbackHandler(
        #to_ignore=["ChannelRead", "RunnableLambda", "ChannelWrite", "__start__", "_execute", "agent", "action", "should_continue", "ChatOpenAI", "_write", "TaskRouter"]
    ),
    langfuse_handler
    ]
}

_printed = set()

def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

async def main():
    print("""Welcome to the NLP-Application for Building Codes and Standards""")
    user_input = input("Your query: ")
    
    # Create a HumanMessage object
    message = HumanMessage(content=user_input)

    async for event in graph.astream({"messages": message}, config=config, interrupt_before=['HumanFeedback'], stream_mode="values"):

    
       #print(event['agent']['messages'][-1].content)
       _print_event(event, _printed)
    snapshot = graph.get_state(config)
    while snapshot.next:
        # Get graph message
        user_answer = input("Your answer: ")

        result = await graph.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content=user_answer,
                        )
                    ]
                },
                config,
            )
        snapshot = graph.get_state(config)



if __name__ == "__main__":
    asyncio.run(main())





