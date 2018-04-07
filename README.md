# Planner for Human-Robot Interaction

In this project, we developed algorithms that enables a mobile robot to actively communicate with human to facilitate navigation.

### Setting up
The project has a few dependencies:
1. [NLopt](https://nlopt.readthedocs.io/en/latest/) - a non-linear optimization library.
2. [ROS People Package](https://github.com/wg-perception/people) - for subscribing to people tracking results. **Note**: has to check out the **kinetic** branch in order to compile in ros-kinetics. However, this will skip the *leg-detector* package.
3. [MediaX package](https://github.com/yuhangc/mediax_project) - in order to use the haptic feedback. Without this the communication message still gets published.
