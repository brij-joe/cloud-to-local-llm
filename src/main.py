import os

from dotenv import load_dotenv
from huggingface_hub import login
from langgraph.graph import START, END, StateGraph
from agents.emotional_agent import emotional_agent
from agents.classifier_agent import classifier_agent
from agents.logical_agent import logical_agent
from utils.state import State

# Load environment variables from tokens.txt file
load_dotenv("C:\\Users\\brij_\\PycharmProjects\\tokens.txt")
login(token=os.environ.get("HF_TOKEN"))


def build_graph():
    builder = StateGraph(State)
    builder.add_node("classifier", classifier_agent)
    builder.add_node("logical", logical_agent)
    builder.add_node("emotional", emotional_agent)

    builder.add_edge(START, "classifier")
    builder.add_conditional_edges("classifier", lambda state: state["classification"],
                                  {"logical": "logical", "emotional": "emotional"})
    builder.add_edge("logical", END)
    builder.add_edge("emotional", END)

    graph = builder.compile()
    return graph


if __name__ == "__main__":
    mygraph = build_graph()
    # Run the graph with some test inputs
    test_inputs = ["Why sky is blue?", "I had a terrible day and I just want to cry.",
                   "Can you explain quantum mechanics in simple terms?",
                   "I'm feeling really anxious about my upcoming exams."]

    for user_input in test_inputs:
        state: State = {'user_input': user_input, 'classification': '', 'response': ''}  # OK

        result = mygraph.invoke(state)
        print(f"\nInput: {result['user_input']}")
        print(f"Classification: {result['classification']}")
        print(f"Response: {result['response']}")
