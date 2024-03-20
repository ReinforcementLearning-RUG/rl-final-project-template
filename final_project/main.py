import gymnasium as gym
from final_project.agents.agentfactory import AgentFactory


def env_interaction(env_str: str, agent_type: str, num_episodes: int) -> None:
    """
    Simulates interaction between an agent and an environment for a given number of episodes.

    :param env_str: The environment string specifying the environment to use.
    :param agent_type: The type of agent to use for interaction.
    :param num_episodes: The number of episodes to simulate interaction for.
    """
    env = gym.make(env_str, render_mode='human')
    obs, info = env.reset()
    agent = AgentFactory.create_agent(agent_type, env=env)

    episode_return: float = 0
    while num_episodes > 0:
        old_obs = obs
        action = agent.policy(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward

        agent.add_trajectory((old_obs, action, reward, obs))
        agent.update()

        if terminated or truncated:
            num_episodes -= 1
            episode_return = 0

            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    env_interaction("InvertedDoublePendulum-v4", 'ACTOR-CRITIC-AGENT', 10)
    env_interaction("InvertedDoublePendulum-v4", 'RANDOM', 10)
