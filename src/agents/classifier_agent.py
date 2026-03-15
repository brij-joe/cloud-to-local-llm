from langchain_core.runnables import RunnableConfig

from config.app_config import MODEL_LLAMA
from utils.state import State
from models.llama_model import LLMModel

def classifier_agent(state: State, config: RunnableConfig) -> State:

    prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Classify the following user input as either 'logical' or 'emotional'. Respond with only one word: logical or emotional. User input:\n\n {state['user_input']}"}
    ]

    model = LLMModel(model_name=MODEL_LLAMA, task_name="text-generation",  max_length=256, temperature=0.7, top_p=0.9)
    classification = model.get_model_response(prompt)

    # Ensure it's one of the two
    if classification not in ["logical", "emotional"]:
        classification = "logical"  # default

    new_state: State = {'user_input': state.get('user_input'), 'classification': classification, 'response': None}  # OK
    return new_state
