from langchain_core.runnables import RunnableConfig

from config.app_config import MODEL_LLAMA
from models.llama_model import LLMModel
from utils.state import State


def logical_agent(state: State, config: RunnableConfig) -> State:
    prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Respond to the following logical query in a rational, analytical manner, and be brief:\n {state['user_input']}"}
    ]
    model = LLMModel(model_name=MODEL_LLAMA, task_name="text-generation", max_length=256, temperature=0.7, top_p=0.9)
    response = model.get_model_response(prompt)
    new_state: State = {'user_input': state.get('user_input'), 'classification': state.get('classification'), 'response': response}  # OK
    return new_state
