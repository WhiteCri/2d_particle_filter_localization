# 2d_particle_filter_localization
Now i understand particle filter...
### state : x, x', y, y', theta, theta'
### Tested with ubuntu 18.04, ros-melodic.

# how to run
simulation
~~~
roslaunch pf_project simulation.launch
~~~
actual 2d grid localization
~~~
roslaunch pf_project particle_filter.launch 
~~~
control robot
~~~
roslaunch turtlebot3_leop turtlebot3_teleop_key.launch
~~~

# requirement
turtlebot3 world
