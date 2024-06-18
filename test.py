#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import gym
import gym_carla
import carla

import textwrap
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.callbacks import OpenAICallbackHandler

import base64
import numpy as np
from PIL import Image
from io import BytesIO


def arr_to_base64(arr):
  birdeye_img = Image.fromarray(arr)
  buffered = BytesIO()
  birdeye_img.save(buffered, format='JPEG')
  image_bytes = buffered.getvalue()
  base64_string = base64.b64encode(image_bytes).decode('utf-8')

  return base64_string


def main():
  # parameters for the gym_carla environment
  params = {
    'number_of_vehicles': 100,
    'number_of_walkers': 0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': False,  # whether to use discrete control space
    'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
    'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
    'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
    'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
    'port': 2000,  # connection port
    'town': 'Town05',  # which town to simulate
    'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': 1000,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 32,  # observation range (meter)
    'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,  # threshold for out of lane
    'desired_speed': 8,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': False,  # whether to output PIXOR observation
  }

  # Set gym-carla environment
  env = gym.make('carla-v0', params=params)
  obs = env.reset()

  # llm related settings
  llm = ChatOpenAI(
    temperature=0.0,
    # top_p=0.0,
    callbacks=[
      OpenAICallbackHandler()
    ],
    model_name='gpt-4-all',
    max_tokens=2000,
    request_timeout=60,
    streaming=True,
    base_url="https://api.132006.xyz/v1"
  )

  while True:
    action = [2.0, 0.0]
    obs,r,done,info = env.step(action)
    # print(obs)

    # test for llm
    birdeye_base64 = arr_to_base64(obs['birdeye'])
    prompt = [
      {'type': 'text', 'text': f'Please describe the traffic scene represented by the Bird-Eye-View image.'},
      {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{birdeye_base64}"},}
    ]
    messages = [HumanMessage(content=prompt)]
    print("Agent answer:")
    response_content = ""
    for chunk in llm.stream(messages):
      response_content += chunk.content
      print(chunk.content, end="", flush=True)
    print("\n")

    env.render()

    if done:
      obs = env.reset()


if __name__ == '__main__':
  main()