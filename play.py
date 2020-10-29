#!/usr/bin/env python3

""" Front-end script for replaying the Robotaxi agent's behavior on a batch of episodes. """

import json
import sys, os
import numpy as np

from robotaxi.gameplay.environment import Environment
from robotaxi.gui import PyGameGUI
from robotaxi.utils.cli import HelpOnFailArgumentParser
from robotaxi.gameplay.entities import CellType


def parse_command_line_args(args):
    """ Parse command-line arguments and organize them into a single structured object. """

    parser = HelpOnFailArgumentParser(
        description='Robotaxi AI replay client.',
        epilog='Example: play.py --agent dqn --model dqn-final.model --level 10x10.json'
    )

    parser.add_argument(
        '--interface',
        type=str,
        choices=['cli', 'gui'],
        default='gui',
        help='Interface mode (command-line or GUI).',
    )
    parser.add_argument(
        '--agent',
        required=True,
        type=str,
        choices=['human', 'dqn', 'random', 'val-itr', 'mixed'],
        help='Player agent to use.',
    )
    parser.add_argument(
        '--model',
        type=str,
        help='File containing a pre-trained agent model.',
    )
    parser.add_argument(
        '--level',
        type=str,
        default='./robotaxi/levels/8x8-blank.json',
        help='JSON file containing a level definition.',
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=1,
        help='The number of episodes to run consecutively.',
    )
    parser.add_argument(
        '--save_frames', 
        action="store_true", 
        default=False, 
        help='save frames as jpg files in screenshots/ folder.'
    )
    parser.add_argument(
        '--stationary', 
        action="store_true", 
        default=False, 
        help='determine whether the environment is stationary'
    )

    parser.add_argument(
        '--collaborating_agent', 
        type=str,
        choices=['human', 'dqn', 'random', 'val-itr', 'mixed'],
        help='Collaborator agent to use.',
    )

    parser.add_argument(
        '--collaborator_model',
        type=str,
        help='File containing a pre-trained agent model.',
    )
    
    parser.add_argument(
        '--participant',
        type=str,
        default='test',
        help='Participant ID.',
    )

    parser.add_argument(
        '--test_run', 
        action="store_true", 
        default=False, 
        help='determine whether the environment is stationary'
    )


    return parser.parse_args(args)


def create_robotaxi_environment(level_filename, stationary, collaboration, test=False, participant=None):
    """ Create a new Robotaxi environment from the config file. """

    with open(level_filename) as cfg:
        env_config = json.load(cfg)
    if test: env_config["max_step_limit"] = 50

    return Environment(config=env_config, stationary=stationary, collaboration=collaboration, verbose=0, participant=participant)


def load_model(filename):
    """ Load a pre-trained agent model. """

    from keras.models import load_model
    return load_model(filename)


def create_agent(name, model, dimension, env, reward_mapping=None):
    """
    Create a specific type of Robotaxi AI agent.
    
    Args:
        name (str): key identifying the agent type.
        model: (optional) a pre-trained model required by certain agents.

    Returns:
        An instance of Robotaxi agent.
    """

    from robotaxi.agent import HumanAgent, RandomActionAgent, ValueIterationAgent, MixedActionAgent, RewardLearningAgent

    if name == 'human':
        return HumanAgent()
    elif name == 'random':
        return RandomActionAgent()
    elif name == 'val-itr':
        return ValueIterationAgent(grid_size=dimension, env=env, reward_mapping=reward_mapping)
    elif name == 'mixed':
        return MixedActionAgent(grid_size=dimension, env=env)
    elif name == 'reward-learning':
        return RewardLearningAgent()
    raise KeyError(f'Unknown agent type: "{name}"')


def play_cli(env, agent, agent_name, num_episodes=1):
    """
    Play a set of episodes using the specified Robotaxi agent.
    Use the non-interactive command-line interface and print the summary statistics afterwards.
    
    Args:
        env: an instance of Robotaxi environment.
        agent: an instance of Robotaxi agent.
        num_episodes (int): the number of episodes to run.
    """

    good_fruit_stats = []
    bad_fruit_stats = []
    lava_stats = []
    score_stats = []

    print()
    print('Playing:')

    print('Episode | Score | Good Fruits | Bad Fruits | Lava ')
    for episode in range(num_episodes):
        timestep = env.new_episode()
        agent.begin_episode()
        game_over = False

        while not game_over:
            action = agent.act(timestep.observation, timestep.reward)
            #print(action)
            env.choose_action(action)
            if agent_name == 'mixed':
                timestep = env.timestep(agent_mode=agent.curr_agent)
            else:
                timestep = env.timestep()
            game_over = timestep.is_episode_end

        good_fruit_stats.append(env.stats.good_fruits_eaten)
        bad_fruit_stats.append(env.stats.bad_fruits_eaten)
        lava_stats.append(env.stats.lava_crossed)
        score_stats.append(env.stats.sum_episode_rewards)

        summary = '{:3d}/{:3d} | {:4.1f}  |   {:3d}   |   {:3d}   | {:3d}'
        print(summary.format(episode + 1, num_episodes, env.stats.sum_episode_rewards, env.stats.good_fruits_eaten, env.stats.bad_fruits_eaten, env.stats.lava_crossed))

    print()
    print('Good Fruits eaten {:.1f} +/- {:.1f}'.format(np.mean(good_fruit_stats), np.std(good_fruit_stats)))
    print('Bad Fruits eaten {:.1f} +/- {:.1f}'.format(np.mean(bad_fruit_stats), np.std(bad_fruit_stats)))
    print('Lava eaten {:.1f} +/- {:.1f}'.format(np.mean(lava_stats), np.std(lava_stats)))
    print('Final Score {:.1f} +/- {:.1f}'.format(np.mean(score_stats), np.std(score_stats)))


def play_gui(env, agent, agent_name, num_episodes, save_frames, field_size, collaborating_agent, collaborating_agent_name, participant, test=False):
    """
    Play a set of episodes using the specified Robotaxi agent.
    Use the interactive graphical interface.
    
    Args:
        env: an instance of Robotaxi environment.
        agent: an instance of Robotaxi agent.
        num_episodes (int): the number of episodes to run.
    """

    gui = PyGameGUI(save_frames=save_frames, field_size=field_size, test=test)
    gui.load_environment(env)
    gui.load_agent(agent, agent_name)
    if collaborating_agent is not None:
        gui.load_collaborator(collaborating_agent, collaborating_agent_name)
    gui.run(num_episodes=num_episodes, participant=participant)
    
    if collaborating_agent is not None:
        print('Final Score {:.1f} '.format(env.stats.sum_episode_rewards+env.stats_collaborator.sum_episode_rewards))

def main():

    if not os.path.exists('./log/'): os.makedirs('./log/') 
    if not os.path.exists('./csv/'): os.makedirs('./csv/') 
    parsed_args = parse_command_line_args(sys.argv[1:])

    collaboration = False if parsed_args.collaborating_agent is None else True
    if collaboration: parsed_args.level = 'robotaxi/levels/8x8-blank-collaboration.json'

    env = create_robotaxi_environment(parsed_args.level, parsed_args.stationary, collaboration, parsed_args.test_run, participant=parsed_args.participant)
    model = load_model(parsed_args.model) if parsed_args.model is not None else None
    dimension = int(parsed_args.level.split('/')[-1].split('x')[0])
    agent = create_agent(parsed_args.agent, model, dimension, env)
    collaborator_model = load_model(parsed_args.collaborator_model) if parsed_args.collaborator_model is not None else None
    reward_mapping = {
                CellType.ROBOTAXI_HEAD: 0,
                CellType.ROBOTAXI_BODY: 0,
                CellType.COLLABORATOR_HEAD: 0,
                CellType.COLLABORATOR_BODY: 0,
                CellType.GOOD_FRUIT: -5,
                CellType.BAD_FRUIT: -1,
                CellType.LAVA: 6,
                CellType.EMPTY: 0,
                CellType.PIT: 0,
                CellType.WALL: -100,
            }
    collaborating_agent = create_agent(parsed_args.collaborating_agent, collaborator_model, dimension, env, reward_mapping=reward_mapping) if collaboration else None

    if parsed_args.interface == 'cli':
        play_cli(env, agent, parsed_args.agent, num_episodes=parsed_args.num_episodes)
    else:
        play_gui(env, agent, parsed_args.agent, num_episodes=parsed_args.num_episodes, save_frames=parsed_args.save_frames, field_size=dimension, collaborating_agent=collaborating_agent, collaborating_agent_name=parsed_args.collaborating_agent, participant=parsed_args.participant, test=parsed_args.test_run)

if __name__ == '__main__':
    main()
