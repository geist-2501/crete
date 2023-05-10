from crete import register_agent
from .dqn_agent import DQNAgent, dqn_training_wrapper, dqn_graphing_wrapper

register_agent(
    agent_id="DQN",
    agent_factory=lambda obs, n_actions, device: DQNAgent(obs, n_actions, device=device),
    graphing_wrapper=dqn_graphing_wrapper,
    training_wrapper=dqn_training_wrapper
)