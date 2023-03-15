import gym
import numpy as np
import cv2


class WarpFrame(gym.ObservationWrapper):
    """
        TODO
    """

    def __init__(self, env):
        """
            Warp frames to 84x84 as done in the Nature paper and later work.
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width),
            dtype=np.uint8
        )

    def observation(self, frame):
        """
            TODO
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None].squeeze()
