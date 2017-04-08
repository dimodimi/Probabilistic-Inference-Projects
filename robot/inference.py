#!/usr/bin/env python
# inference.py
# Base code by George H. Chen (georgehc@mit.edu) -- updated 10/18/2016
import collections
import sys

import graphics
import numpy as np
import robot


# Throughout the code, we use these variables.
# Do NOT change these (but you'll need to use them!):
# - all_possible_hidden_states: a list of possible hidden states
# - all_possible_observed_states: a list of possible observed states
# - prior_distribution: a distribution over states
# - transition_model: a function that takes a hidden state and returns a
#     Distribution for the next state
# - observation_model: a function that takes a hidden state and returns a
#     Distribution for the observation from that hidden state
all_possible_hidden_states = robot.get_all_hidden_states()
all_possible_observed_states = robot.get_all_observed_states()
prior_distribution = robot.initial_distribution()
transition_model = robot.transition_model
observation_model = robot.observation_model


# You may find this function helpful for computing logs without yielding a
# NumPy warning when taking the log of 0.
def careful_log(x):
    # computes the log of a non-negative real number
    if x == 0:
        return -np.inf
    else:
        return np.log(x)


# -----------------------------------------------------------------------------
# Functions for you to implement
#

def forward_backward(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    robot.py and see how it is used in both robot.py and the function
    generate_data() above, and the i-th Distribution should correspond to time
    step i
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #
    num_states = len(all_possible_hidden_states)
    num_time_steps = len(observations)
    
    #Create transition matrix A
    A = np.zeros([num_states]*2)
    state_index = dict(zip(all_possible_hidden_states, range(num_states)))
    
    for state in all_possible_hidden_states:
        next_states = transition_model(state)
        for new_state in next_states.keys():
            A[state_index[state], state_index[new_state]] += next_states[new_state]
    
    #Create observation matrix B
    B = np.zeros([num_states, len(all_possible_observed_states)])
    obsv_index = dict(zip(all_possible_observed_states, range(len(all_possible_observed_states))))
    
    for state in all_possible_hidden_states:
        obsv = observation_model(state)
        for tile in obsv.keys():
            B[state_index[state], obsv_index[tile]] += obsv[tile]    
    
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    #TODO: Compute the forward messages
    for i in range(1, num_time_steps):
        dist = [0] * num_states
        if observations[i-1] == None:
            for state in all_possible_hidden_states:
                temp = forward_messages[i-1][state]
                dist += temp * A[state_index[state], :]
        else:
            for state in all_possible_hidden_states:
                temp = forward_messages[i-1][state] * B[state_index[state], obsv_index[observations[i-1]]]
                dist += temp * A[state_index[state], :]
        forward_messages[i] = robot.Distribution()
        for state in all_possible_hidden_states:
            if dist[state_index[state]] > 0:
                forward_messages[i][state] = dist[state_index[state]]

    backward_messages = [None] * num_time_steps
    backward_messages[-1] = robot.Distribution()
    for state in all_possible_hidden_states:
        backward_messages[-1][state] = 1./num_states
    #TODO: Compute the backward messages
    for i in range(num_time_steps-2, -1, -1):
        dist = [0] * num_states
        if observations[i+1] == None:
            for state in all_possible_hidden_states:
                temp = backward_messages[i+1][state]
                dist += temp * A[:, state_index[state]]
        else:
            for state in all_possible_hidden_states:
                temp = backward_messages[i+1][state] * B[state_index[state], obsv_index[observations[i+1]]]
                dist += temp * A[:, state_index[state]]
        backward_messages[i] = robot.Distribution()
        for state in all_possible_hidden_states:
            if dist[state_index[state]] > 0:
                backward_messages[i][state] = dist[state_index[state]]

    marginals = [None] * num_time_steps
    for i in range(num_time_steps):
        marginals[i] = robot.Distribution()
        for state in all_possible_hidden_states:
            if observations[i] == None:
                temp = forward_messages[i][state] * backward_messages[i][state]
            else:
                temp = forward_messages[i][state] * backward_messages[i][state] * B[state_index[state], obsv_index[observations[i]]]
            if temp > 0:
                marginals[i][state] = temp
        marginals[i].renormalize()

    return marginals


def Viterbi(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #

    num_time_steps = len(observations)
    num_states = len(all_possible_hidden_states)
    l2 = np.vectorize(lambda x: -np.log2(x) if x != 0 else np.inf)

    #Create transition matrix A
    A = np.zeros([num_states]*2)
    state_index = dict(zip(all_possible_hidden_states, range(num_states)))
    
    for state in all_possible_hidden_states:
        next_states = transition_model(state)
        for new_state in next_states.keys():
            A[state_index[state], state_index[new_state]] += next_states[new_state]

    A = l2(A)
    
    #Create observation matrix B
    B = np.zeros([num_states, len(all_possible_observed_states)])
    obsv_index = dict(zip(all_possible_observed_states, range(len(all_possible_observed_states))))
    
    for state in all_possible_hidden_states:
        obsv = observation_model(state)
        for tile in obsv.keys():
            B[state_index[state], obsv_index[tile]] += obsv[tile]

    B = l2(B)

    traceback = np.zeros([num_states, num_time_steps-1])
    messages  = np.zeros([num_states, num_time_steps-1])

    phi = []
    for state in all_possible_hidden_states:
        phi.append(prior_distribution[state])
    phi = l2(phi)

    if observations[0] == None:
        messages[:, 0]  = np.amin( A + phi[:, np.newaxis], axis = 0)
        traceback[:, 0] = np.argmin(  A + phi[:, np.newaxis], axis = 0 )
    else:
        messages[:, 0]  = np.amin(  A + B[:, obsv_index[observations[0]]][:, np.newaxis] + phi[:, np.newaxis] , axis = 0 )
        traceback[:, 0] = np.argmin(  A + B[:, obsv_index[observations[0]]][:, np.newaxis] + phi[:, np.newaxis] , axis = 0 )

    for i in range(1, num_time_steps-1):
        if observations[i] == None:
            messages[:, i]  = np.amin( A + messages[:, i-1][:, np.newaxis], axis = 0 )
            traceback[:, i] = np.argmin( A + messages[:, i-1][:, np.newaxis], axis = 0 )
        else:
            messages[:, i]  = np.amin( A + messages[:, i-1][:, np.newaxis] + B[:, obsv_index[observations[i]]][:, np.newaxis], axis = 0 )
            traceback[:, i] = np.argmin( A + messages[:, i-1][:, np.newaxis] + B[:, obsv_index[observations[i]]][:, np.newaxis], axis = 0 )

    index_state = dict(zip(range(num_states), all_possible_hidden_states))
    estimated_hidden_states = [0] * num_time_steps # remove this
    if observations[-1] == None:
        x_hat = np.argmin( messages[:, -1] )
    else:
        x_hat = np.argmin( B[:, obsv_index[observations[-1]]] + messages[:, -1] )
    estimated_hidden_states[-1] = index_state[x_hat]

    for i in range(num_time_steps-2, -1, -1):
        estimated_hidden_states[i] = index_state[traceback[x_hat, i]]
        x_hat = traceback[x_hat, i].astype(int)

    return estimated_hidden_states


def second_best(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #


    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this

    return estimated_hidden_states


# -----------------------------------------------------------------------------
# Generating data from the hidden Markov model
#

def generate_data(num_time_steps, make_some_observations_missing=False,
                  random_seed=None):
    # generate samples from this project's hidden Markov model
    hidden_states = []
    observations = []

    # if the random seed is not None, then this makes the randomness
    # deterministic, which may be helpful for debug purposes
    np.random.seed(random_seed)

    # draw initial state and emit an observation
    initial_state = prior_distribution.sample()
    initial_observation = observation_model(initial_state).sample()

    hidden_states.append(initial_state)
    observations.append(initial_observation)

    for time_step in range(1, num_time_steps):
        # move the robot
        prev_state = hidden_states[-1]
        new_state = transition_model(prev_state).sample()

        # maybe emit an observation
        if not make_some_observations_missing:
            new_observation = observation_model(new_state).sample()
        else:
            if np.random.rand() < .1:  # 0.1 prob. of observation being missing
                new_observation = None
            else:
                new_observation = observation_model(new_state).sample()

        hidden_states.append(new_state)
        observations.append(new_observation)

    return hidden_states, observations


# -----------------------------------------------------------------------------
# Main
#

def main():
    # flags
    make_some_observations_missing = False
    use_graphics = True
    need_to_generate_data = True

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '--missing':
            make_some_observations_missing = True
        elif arg == '--nographics':
            use_graphics = False
        elif arg.startswith('--load='):
            filename = arg[7:]
            hidden_states, observations = robot.load_data(filename)
            need_to_generate_data = False
            num_time_steps = len(hidden_states)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 100
        hidden_states, observations = \
            generate_data(num_time_steps,
                          make_some_observations_missing)

    print('Running forward-backward...')
    marginals = forward_backward(observations)
    print("\n")

    timestep = 99
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        print(sorted(marginals[timestep].items(),
                     key=lambda x: x[1],
                     reverse=True)[:10])
    else:
        print('*No marginal computed*')
    print("\n")

    print('Running Viterbi...')
    estimated_states = Viterbi(observations)
    print("\n")

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states[time_step] is None:
            print('Missing')
        else:
            print(estimated_states[time_step])
    print("\n")

    print('Finding second-best MAP estimate...')
    estimated_states2 = second_best(observations)
    print("\n")

    print("Last 10 hidden states in the second-best MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states2[time_step] is None:
            print('Missing')
        else:
            print(estimated_states2[time_step])
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP estimate and true hidden " +
          "states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states2[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between second-best MAP estimate and " +
          "true hidden states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != estimated_states2[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP and second-best MAP " +
          "estimates:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    # display
    if use_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()


if __name__ == '__main__':
    main()
