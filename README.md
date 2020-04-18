# ReinforceBot
A reinforcement learning toolkit for automating tasks on desktop applications.

Final year project for Computer Science at UWE.

## Overview
There are two main components of this project: the desktop client, and the web
services. Located in `client` and `web` respectively.

The desktop client is a PyGTK application that enables users to create and
train their own RL agents based on recorded experience with other desktop
applications. The web services provide utilities for saving agent parameter
backups and offloading taxing compute to cloud compute runners.
