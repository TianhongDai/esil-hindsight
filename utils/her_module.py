import numpy as np
import copy

"""
her module - change the trajectory of the entire episode

"""
class her_module:
    def __init__(self, reward_func):
        self.reward_func = reward_func

    # update the trajecotry
    def update_trajectory(self, goal, achieved_goal):
        """
        in the case we use the final strategy

        """
        # copy the trajectory
        goal_her = copy.deepcopy(goal)
        # replace the goal to the last achieved goal
        for i in range(goal_her.shape[0]):
            goal_her[i, :, :] = achieved_goal[i, -1, :]
        # we need to re calculate the rewards for each trajectory
        rewards = np.zeros((goal.shape[0], goal.shape[1]))
        for i in range(goal.shape[0]):
            rewards[i] = self.reward_func(achieved_goal[i, 1:, :], goal_her[i, :, :], None)
        return goal_her, rewards
