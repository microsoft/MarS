#!/usr/bash

# stop existing Ray cluster
ray stop

# build and start the model serving app
# serve build market_simulation.rollout.order_moder_serving:order_model_app -o ray_serving.yaml
ray start --head
serve deploy ray_serving.yaml
serve status
