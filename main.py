import os
from dotenv import load_dotenv
from groq import Groq
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

MODEL= "llama-3.1-8b-instant"

load_dotenv()

# Define state schema
class State(TypedDict):
    topic: str        # properties of the class State which will be passed via agents
    research: str
    draft: str
    review_result: str

# Initialize client globally so all functions can access it
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def researcher_node(state: State) -> State:
    topic = state.get("topic", "Artificial Intelligence")  # Default topic if not provided
    prompt = f"Please research and provide 3 key facts about: {topic}"
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model=MODEL,
        temperature=0.4  # temperature controls the randomness
    )
    state["research"] = chat_completion.choices[0].message.content
    return state

def writer_node(state: State) -> State:
    research = state["research"]  # research will receive research topic
    prompt = f"Based on this research:\n{research}\nWrite a short paragraph (3-5 sentences) suitable for an article."

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model=MODEL,
        temperature=0.7
    )

    state["draft"] = chat_completion.choices[0].message.content
    return state

def reviewer_node(state: State) -> State:
    draft = state["draft"]
    prompt = f"Review this writing paragraph and reply with 'approved' or 'needs revision':\n{draft}"
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model=MODEL,
        temperature=0.3
    )

    state["review_result"] = chat_completion.choices[0].message.content  # Fixed: should be review_result, not draft
    return state

def route_reviewer(state: State):  # control node
    result = state.get("review_result", "")  # based on review_result 
    if "approved" in result.lower():  # Added .lower() for case-insensitive comparison
        return END   # end the process if approved
    return "researcher"  # pass back to researcher if not approved

def main():
    # Create StateGraph (class) State
    builder = StateGraph(State)  # call constructor and initiate StateGraph
    # passing class State with properties
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_node("reviewer", reviewer_node)

    builder.set_entry_point("researcher")  # setting the first node
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", "reviewer")
    builder.add_conditional_edges("reviewer", route_reviewer) # conditional edge based on review result if approved ends or gets back to researcher

    graph = builder.compile()

    # Run the graph with a user topic
    initial_state = {"topic": "Climate Change"} # if you remove "topic": "Climate Change" here, topic will default to "Artificial Intelligence"
    final_state = graph.invoke(initial_state)

    # Output final results
    print("\n=== Final Output ===")
    print(f"Research:\n{final_state['research']}")
    print(f"\nDraft:\n{final_state['draft']}")
    print(f"\nReview Status: {final_state['review_result'].upper()}")

if __name__ == "__main__":
    main()