Terminal1:
```
docker-compose up --build
```

```
ARNING: Found orphan containers (a55a8ae7a762_docker_my_cuda_service_1) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Building my-service
[+] Building 1.6s (10/10) FINISHED                                                                       docker:default
 => [internal] load build definition from Dockerfile                                                               0.2s
 => => transferring dockerfile: 348B                                                                               0.0s
 => [internal] load metadata for docker.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04                             0.0s
 => [internal] load .dockerignore                                                                                  0.2s
 => => transferring context: 2B                                                                                    0.0s
 => [1/5] FROM docker.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04                                               0.0s
 => [internal] load build context                                                                                  0.2s
 => => transferring context: 34B                                                                                   0.0s
 => CACHED [2/5] WORKDIR /app                                                                                      0.0s
 => CACHED [3/5] COPY sources.list /etc/apt/sources.list                                                           0.0s
 => CACHED [4/5] RUN apt update                                                                                    0.0s
 => CACHED [5/5] WORKDIR  /app/sim/                                                                                0.0s
 => exporting to image                                                                                             0.2s
 => => exporting layers                                                                                            0.0s
 => => writing image sha256:8e9c02e6c8b50dcbf7d6d1962d51de926126f132b65b65952ad8dfc74634f8c6                       0.0s
 => => naming to docker.io/library/docker_my-service                                                               0.1s
WARNING: Image for service my-service was built because it did not already exist. To rebuild this image you must use `docker-compose build` or `docker-compose up --build`.
Creating docker_my-service_1 ... done
Attaching to docker_my-service_1
my-service_1  | 
my-service_1  | ==========
my-service_1  | == CUDA ==
my-service_1  | ==========
my-service_1  | 
my-service_1  | CUDA Version 12.1.0
my-service_1  | 
my-service_1  | Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
my-service_1  | 
my-service_1  | This container image and its contents are governed by the NVIDIA Deep Learning Container License.
my-service_1  | By pulling and using the container, you accept the terms and conditions of this license:
my-service_1  | https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
my-service_1  | 
my-service_1  | A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
my-service_1  | 
my-service_1  | *************************
my-service_1  | ** DEPRECATION NOTICE! **
my-service_1  | *************************
my-service_1  | THIS IMAGE IS DEPRECATED and is scheduled for DELETION.
my-service_1  |     https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/support-policy.md
my-service_1  | 
```

Terminal2:

```
docker exec -it docker_zeroth-sim_1 /bin/bash
```

```
gymuser@06aac36e0751:/app/sim# python3 sim/train.py --task=stompymicro --num_envs=4
```