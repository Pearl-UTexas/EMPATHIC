#!/usr/bin/env python3

""" Front-end script for replaying the Robotaxi agent's behavior on a batch of episodes. """

import json
import sys, os
import numpy as np

import os
sys.path.insert(1, "/home/yuchen/projects/RoboTaxiEnv")

from robotaxi.gameplay.environment import Environment
from robotaxi.gui import PyGameGUI
from robotaxi.utils.cli import HelpOnFailArgumentParser
from robotaxi.gameplay.entities import CellType
import subprocess
import signal

def parse_command_line_args(args):
    """ Parse command-line arguments and organize them into a single structured object. """

    parser = HelpOnFailArgumentParser(
        description='Robotaxi AI replay client.',
        epilog='Example: play.py --agent dqn --model dqn-final.model --level 10x10.json'
    )

    parser.add_argument(
        '--interface',
        type=str,
        choices=['cli' 'gui'],
        default='gui',
        help='Interface mode (command-line or GUI).',
    )
    parser.add_argument(
        '--agent',
        type=str,
        choices=['human', 'random', 'val-itr', 'reward-learning', 'mixed'],
        default='reward-learning',
        help='Player agent to use.',
    )

    parser.add_argument(
        '--level',
        type=str,
        default='./RoboTaxiEnv/robotaxi/levels/8x8-blank.json',
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
        '--participant',
        type=str,
        default='test',
        help='Participant ID.',
    )

    parser.add_argument(
        '--test_run', 
        action="store_true", 
        default=False, 
        help='determine whether this is a test run'
    )

    return parser.parse_args(args)


def create_robotaxi_environment(level_filename, stationary, collaboration, test=False, participant=None):
    """ Create a new Robotaxi environment from the config file. """

    with open(level_filename) as cfg:
        env_config = json.load(cfg)
    if test: env_config["max_step_limit"] = 50

    return Environment(config=env_config, stationary=stationary, collaboration=collaboration, verbose=2, participant=participant)


def load_model(filename):
    """ Load a pre-trained agent model. """

    from keras.models import load_model
    return load_model(filename)


def create_agent(name, model, dimension, env, participant=None, reward_mapping=None):
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
        return RewardLearningAgent(participant)
   
    raise KeyError(f'Unknown agent type: "{name}"')


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
    
    gui.run(num_episodes=num_episodes, participant=participant, online_learning=True)
    
   
def main():

    if not os.path.exists('./log/'): os.makedirs('./log/') 
    if not os.path.exists('./csv/'): os.makedirs('./csv/') 
    if not os.path.exists('./openface_out/'): os.makedirs('openface_out/') 

    parsed_args = parse_command_line_args(sys.argv[1:])
    env = create_robotaxi_environment(parsed_args.level, parsed_args.stationary, False, parsed_args.test_run, participant=parsed_args.participant)
    model = None
    collaborator_model = None
    collaborating_agent =  None
    dimension = int(parsed_args.level.split('/')[-1].split('x')[0])
    #agent = create_agent(parsed_args.agent, model, dimension, env)   
    agent = create_agent(parsed_args.agent, model, dimension, env, participant=parsed_args.participant)
    
    #proc = subprocess.Popen(['/bin/bash','start_openface.bash'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    proc = subprocess.Popen(['/bin/bash','start_openface.bash',parsed_args.participant], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    play_gui(env, agent, parsed_args.agent, num_episodes=parsed_args.num_episodes, save_frames=parsed_args.save_frames, field_size=dimension, collaborating_agent=collaborating_agent, collaborating_agent_name=collaborating_agent, participant=parsed_args.participant, test=parsed_args.test_run)
    proc.stdout.close()
    proc.wait()    
    try:      
        proc.kill()  
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM) 
    finally:
        sys.exit()

if __name__ == '__main__':
    main()
