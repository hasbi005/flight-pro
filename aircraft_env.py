
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from delay_predictor import predict_airport_delay


class AircraftReallocationEnv(gym.Env):
    """
    Simulated aircraft reallocation environment
    using LSTM-predicted airport delay.
    """

    def __init__(self):
        super().__init__()

        # Load an LSTM prediction once.
        self.predicted_delay = float(
            np.clip(predict_airport_delay(), 0, 180)
        )

        # Actions:
        # 0 = Keep original aircraft
        # 1 = Reallocate spare aircraft
        # 2 = Hold flight temporarily
        self.action_space = spaces.Discrete(3)

        # Observation:
        # [predicted_delay, disruption, spare_aircraft, availability]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        self.disruption = 0
        self.spare_aircraft = 0
        self.availability = 1

    def _get_observation(self):
        return np.array(
            [
                self.predicted_delay / 180.0,
                float(self.disruption),
                float(self.spare_aircraft),
                float(self.availability),
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Simulate a disruption scenario.
        self.disruption = int(
            self.np_random.random() < 0.35
        )

        self.spare_aircraft = int(
            self.np_random.random() < 0.60
        )

        self.availability = 1 - self.disruption

        return self._get_observation(), {}

    def step(self, action):
        final_delay = self.predicted_delay
        reward = 0.0
        reallocated = False

        if action == 0:
            # Keep original aircraft.
            reward = -final_delay / 60.0

            if self.disruption:
                reward -= 2.0

        elif action == 1:
            # Reallocate a spare aircraft.
            if self.spare_aircraft:
                reallocated = True

                final_delay = max(
                    0.0,
                    self.predicted_delay - 45.0,
                )

                reward = -final_delay / 60.0 - 0.5

                if not self.disruption:
                    reward -= 1.0
            else:
                reward = -5.0

        elif action == 2:
            # Temporarily hold the flight.
            final_delay = min(
                180.0,
                self.predicted_delay + 15.0,
            )

            reward = -final_delay / 60.0 - 0.5

        else:
            raise ValueError("Invalid action")

        info = {
            "predicted_delay_minutes": round(
                self.predicted_delay, 2
            ),
            "final_delay_minutes": round(
                final_delay, 2
            ),
            "disruption": self.disruption,
            "spare_aircraft": self.spare_aircraft,
            "action": int(action),
            "reallocated": reallocated,
        }

        terminated = True
        truncated = False

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            info,
        )
