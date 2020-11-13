import numpy as np
import cv2
import pygame
import os
from shapely.geometry import LineString


class CarTrackSimulator:
    def __init__(self, visualize_enable):
        self.observation_space = 5
        self.action_space = 2
        self.action_space_low = np.array([-10, -np.pi/40])
        self.action_space_high = np.array([10, np.pi/40])
        self.visualize_enable = visualize_enable
        self.width = 1000
        self.height = 700
        self.Race_track_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "Car_track_model", "track_0.2.png")
        self.background = cv2.imread(self.Race_track_path, cv2.IMREAD_COLOR)
        self.frame = self.background.copy()
        self.checkpoint_lines = np.array([[(63, 307), (249, 306)], [(108, 126), (250, 271)], [(255, 79), (286, 244)],
                                          [(369, 84), (379, 252)], [(459, 88), (478, 261)], [(553, 85), (579, 288)],
                                          [(630, 79), (690, 285)], [(771, 64), (756, 286)], [(940, 117), (783, 285)],
                                          [(981, 301), (772, 339)], [(964, 489), (766, 382)], [(928, 603), (714, 441)],
                                          [(669, 630), (657, 442)], [(543, 414), (546, 610)], [(444, 391), (418, 588)],
                                          [(328, 370), (303, 577)], [(258, 361), (138, 574)], [(246, 343), (49, 447)]],
                                         dtype=np.uint32)
        self.checkpoint_center = np.zeros((self.checkpoint_lines.shape[0], 2), dtype=np.uint32)
        for i in range(self.checkpoint_lines.shape[0]):
            self.checkpoint_center[i][0] = (self.checkpoint_lines[i][0][0] + self.checkpoint_lines[i][1][0]) / 2
            self.checkpoint_center[i][1] = (self.checkpoint_lines[i][0][1] + self.checkpoint_lines[i][1][1]) / 2
        self.car_1 = self.Car(visualize_enable=self.visualize_enable)
        if self.visualize_enable:
            self.init_pygame()
            sensor_1_output, sensor_1_line = self.car_1.get_sensor_output(self, self.car_1.rotate_angle)
            sensor_2_output, sensor_2_line = self.car_1.get_sensor_output(self, self.car_1.rotate_angle
                                                                          + np.pi / 4)
            sensor_3_output, sensor_3_line = self.car_1.get_sensor_output(self, self.car_1.rotate_angle
                                                                          - np.pi / 4)
            sensor_4_output, sensor_4_line = self.car_1.get_sensor_output(self, self.car_1.rotate_angle
                                                                          + np.pi * 3.2 / 4)
            sensor_5_output, sensor_5_line = self.car_1.get_sensor_output(self, self.car_1.rotate_angle
                                                                          - np.pi * 3.2 / 4)
            sensors_output = np.array([sensor_1_output, sensor_2_output,
                                       sensor_3_output, sensor_4_output, sensor_5_output])
            sensors_lines = np.array([sensor_1_line, sensor_2_line, sensor_3_line, sensor_4_line, sensor_5_line])
            self.pre_sensors_output = sensors_output
            self.pre_sensors_lines = sensors_lines
        else:
            sensor_1_output = self.car_1.get_sensor_output(self, self.car_1.rotate_angle)
            sensor_2_output = self.car_1.get_sensor_output(self, self.car_1.rotate_angle + np.pi / 4)
            sensor_3_output = self.car_1.get_sensor_output(self, self.car_1.rotate_angle - np.pi / 4)
            sensor_4_output = self.car_1.get_sensor_output(self, self.car_1.rotate_angle + np.pi * 3.2 / 4)
            sensor_5_output = self.car_1.get_sensor_output(self, self.car_1.rotate_angle - np.pi * 3.2 / 4)
            sensors_output = np.array([sensor_1_output, sensor_2_output,
                                       sensor_3_output, sensor_4_output, sensor_5_output])
            self.pre_sensors_output = sensors_output

    def step(self, velocity, rotate_angle_speed):
        velocity = np.clip(velocity, self.action_space_low[0], self.action_space_high[0])
        rotate_angle_speed = np.clip(rotate_angle_speed, self.action_space_low[1], self.action_space_high[1])

        self.car_1.check_border_intersect(self.background, self.frame)
        self.car_1.update_coords(rotate_angle_speed, velocity)
        if self.car_1.is_dead != 1:
            reward = self.car_1.get_reward(self)
        self.draw_cars(self.car_1)
        if self.visualize_enable:
            if not self.car_1.is_dead:
                sensor_1_output, sensor_1_line = self.car_1.get_sensor_output(self, self.car_1.rotate_angle)
                sensor_2_output, sensor_2_line = self.car_1.get_sensor_output(self, self.car_1.rotate_angle
                                                                              + np.pi / 4)
                sensor_3_output, sensor_3_line = self.car_1.get_sensor_output(self, self.car_1.rotate_angle
                                                                              - np.pi / 4)
                sensor_4_output, sensor_4_line = self.car_1.get_sensor_output(self, self.car_1.rotate_angle
                                                                              + np.pi * 3.2 / 4)
                sensor_5_output, sensor_5_line = self.car_1.get_sensor_output(self, self.car_1.rotate_angle
                                                                              - np.pi * 3.2 / 4)
                sensors_output = np.array([sensor_1_output, sensor_2_output,
                                           sensor_3_output, sensor_4_output, sensor_5_output])
                sensors_lines = np.array([sensor_1_line, sensor_2_line, sensor_3_line, sensor_4_line, sensor_5_line])
                if min(sensor_1_output, sensor_2_output, sensor_3_output, sensor_4_output, sensor_5_output) < 5:
                    sensors_output = self.pre_sensors_output
                    sensors_lines = self.pre_sensors_lines
                    self.car_1.total_reward -= self.car_1.dead_penalty
                    self.car_1.is_dead = 1
                else:
                    self.pre_sensors_output = sensors_output
                    self.pre_sensors_lines = sensors_lines
            else:
                sensors_output = self.pre_sensors_output
                sensors_lines = self.pre_sensors_lines
            self.update_visualize(sensors_output, sensors_lines)
        else:
            if not self.car_1.is_dead:
                sensor_1_output = self.car_1.get_sensor_output(self, self.car_1.rotate_angle)
                sensor_2_output = self.car_1.get_sensor_output(self, self.car_1.rotate_angle
                                                               + np.pi / 4)
                sensor_3_output = self.car_1.get_sensor_output(self, self.car_1.rotate_angle
                                                               - np.pi / 4)
                sensor_4_output = self.car_1.get_sensor_output(self, self.car_1.rotate_angle
                                                               + np.pi * 3.2 / 4)
                sensor_5_output = self.car_1.get_sensor_output(self, self.car_1.rotate_angle
                                                               - np.pi * 3.2 / 4)
                sensors_output = np.array([sensor_1_output, sensor_2_output,
                                           sensor_3_output, sensor_4_output, sensor_5_output])
                if min(sensor_1_output, sensor_2_output, sensor_3_output, sensor_4_output, sensor_5_output) < 5:
                    sensors_output = self.pre_sensors_output
                    self.car_1.total_reward -= self.car_1.dead_penalty
                    self.car_1.is_dead = 1
                else:
                    self.pre_sensors_output = sensors_output
            else:
                sensors_output = self.pre_sensors_output

        return sensors_output, self.car_1.total_reward, self.car_1.is_dead

    class Car:
        def __init__(self, visualize_enable):
            self.checkpoint_reward = 0
            self.checkpoint_reward_step = 10
            self.dead_penalty = 100
            self.total_reward = 0
            self.visualize = visualize_enable
            self.width = np.float(20)
            self.height = np.float(10)
            self.diagonal = np.sqrt((self.width / 2) ** 2 + (self.height / 2) ** 2)
            self.alpha = np.arctan(self.height / self.width)
            self.sin_alpha = np.sin(self.alpha)
            self.cos_alpha = np.cos(self.alpha)
            self.center_coord = np.array([120, 350], np.float)
            self.rotate_angle = np.float(-np.pi / 2)
            self.coordinates_int = (self.diagonal * np.array(
                [[self.cos_alpha, self.sin_alpha],
                 [self.cos_alpha, -self.sin_alpha],
                 [-self.cos_alpha, -self.sin_alpha],
                 [-self.cos_alpha, self.sin_alpha]],
                np.float) + np.repeat([self.center_coord], 4, axis=0) + 0.5).astype(np.uint)
            # self.pre_coordinates_int = self.coordinates_int
            self.is_dead = 0
            self.last_checkpoint = 17

        def update_coords(self, rotate_angle_speed_ldf, speed_ldu):
            if self.is_dead == 0:
                # self.pre_coordinates_int = self.coordinates_int
                self.center_coord += np.array([speed_ldu * np.cos(self.rotate_angle),
                                               speed_ldu * np.sin(self.rotate_angle)],
                                              np.float)
                self.center_coord[0] = max((min(self.center_coord[0], np.uint(995 - max(self.width, self.height) / 2))),
                                           np.uint(max(self.width, self.height) / 2 + 5))
                self.center_coord[1] = max((min(self.center_coord[1], np.uint(695 - max(self.width, self.height) / 2))),
                                           np.uint(max(self.width, self.height) / 2 + 5))

                if self.rotate_angle >= 2 * np.pi:
                    self.rotate_angle -= 2 * np.pi
                elif self.rotate_angle <= -2 * np.pi:
                    self.rotate_angle += 2 * np.pi

                self.rotate_angle += rotate_angle_speed_ldf
                sin_rotate = np.sin(self.rotate_angle)
                cos_rotate = np.cos(self.rotate_angle)
                cos_alpha_plus_rotate = self.cos_alpha * cos_rotate - self.sin_alpha * sin_rotate
                sin_alpha_plus_rotate = self.sin_alpha * cos_rotate + self.cos_alpha * sin_rotate
                cos_alpha_minus_rotate = self.cos_alpha * cos_rotate + self.sin_alpha * sin_rotate
                sin_alpha_minus_rotate = self.sin_alpha * cos_rotate - self.cos_alpha * sin_rotate

                self.coordinates_int = (self.diagonal * np.array(
                    [[cos_alpha_plus_rotate, sin_alpha_plus_rotate],
                     [cos_alpha_minus_rotate, -sin_alpha_minus_rotate],
                     [-cos_alpha_plus_rotate, -sin_alpha_plus_rotate],
                     [-cos_alpha_minus_rotate, sin_alpha_minus_rotate]],
                    np.float) + np.repeat([self.center_coord], 4, axis=0) + 0.5).astype(np.uint)
            else:
                pass

        def check_border_intersect(self, background, frame):
            start_x = np.min(self.coordinates_int[:, 0])
            start_y = np.min(self.coordinates_int[:, 1])
            end_x = np.max(self.coordinates_int[:, 0])
            end_y = np.max(self.coordinates_int[:, 1])
            for x in np.arange(start_x, end_x, 1):
                for y in np.arange(start_y, end_y, 1):
                    if (background[int(y), int(x)] == (0, 0, 0)).all() and \
                            (frame[int(y), int(x)] == (0, 0, 255)).all():
                        if self.is_dead == 0:
                            self.total_reward -= self.dead_penalty
                        self.is_dead = 1

        def get_sensor_output(self, env, sensor_angle):
            line_array = self.get_sensor_line(env.background.shape, sensor_angle)
            distance = -1
            sensor_range = np.array([[line_array[0, 0], line_array[1, 0]], [line_array[0, -1], line_array[1, -1]]])
            for index in range(line_array.shape[1]):
                if (env.background[line_array[1, index], line_array[0, index]] == (0, 0, 0)).all():
                    distance = np.sqrt((self.center_coord[1] +
                                        np.sin(self.rotate_angle) * self.diagonal - line_array[1, index]) ** 2 +
                                       (self.center_coord[0] +
                                        np.cos(self.rotate_angle) * self.diagonal - line_array[0, index]) ** 2)
                    if self.visualize:
                        sensor_range[1][0] = line_array[0, index]
                        sensor_range[1][1] = line_array[1, index]
                        return distance, sensor_range
                    else:
                        return distance
            if self.visualize:
                return distance, np.array([-1])
            else:
                return distance

        def get_sensor_line(self, frame_size, sensor_angle):
            cos_sensor_angle = np.cos(sensor_angle)
            sin_sensor_angle = np.sin(sensor_angle)
            if cos_sensor_angle == 0:
                t = np.arange(0, (frame_size[0] - 5 - self.center_coord[1]) / sin_sensor_angle, 1 / sin_sensor_angle)
            elif sin_sensor_angle == 0:
                t = np.arange(0, (frame_size[1] - 5 - self.center_coord[0]) / cos_sensor_angle, 1 / cos_sensor_angle)
            elif sin_sensor_angle > 0:
                if cos_sensor_angle > 0:
                    t = np.arange(0, min((frame_size[1] - 5 - self.center_coord[0]) / cos_sensor_angle,
                                         (frame_size[0] - 5 - self.center_coord[1]) / sin_sensor_angle),
                                  min(1 / cos_sensor_angle, 1 / sin_sensor_angle))
                else:
                    t = np.arange(0, min(-(self.center_coord[0] - 5) / cos_sensor_angle,
                                         (frame_size[0] - 5 - self.center_coord[1]) / sin_sensor_angle),
                                  min(abs(1 / cos_sensor_angle), abs(1 / sin_sensor_angle)))
            else:
                if cos_sensor_angle > 0:
                    t = np.arange(0, min(-(self.center_coord[1] - 5) / sin_sensor_angle,
                                         (frame_size[1] - 5 - self.center_coord[0]) / cos_sensor_angle),
                                  min(abs(1 / cos_sensor_angle), abs(1 / sin_sensor_angle)))
                else:
                    t = np.arange(0, min(-(self.center_coord[0] - 5) / cos_sensor_angle,
                                         -(self.center_coord[1] - 5) / sin_sensor_angle),
                                  min(abs(1 / cos_sensor_angle), abs(1 / sin_sensor_angle)))
            line_array = np.array([np.clip(self.center_coord[0] + self.width / 2 * np.cos(self.rotate_angle) +
                                           t * cos_sensor_angle, 0, frame_size[1] - 1),
                                   np.clip(self.center_coord[1] + self.width / 2 * np.sin(self.rotate_angle) +
                                           t * sin_sensor_angle, 0, frame_size[0] - 1)], np.uint)
            return line_array

        def get_reward(self, env):
            checkpoint_intersected = -1
            for i in range(env.checkpoint_lines.shape[0]):
                checkpoint = LineString(env.checkpoint_lines[i])
                car_edge = LineString([self.coordinates_int[1], self.coordinates_int[2]])
                if not checkpoint.intersects(car_edge):
                    pass
                else:
                    checkpoint_intersected = i
            if checkpoint_intersected != -1:
                if checkpoint_intersected != self.last_checkpoint:
                    if self.last_checkpoint == 17 and checkpoint_intersected == 0:
                        self.checkpoint_reward += self.checkpoint_reward_step
                    elif self.last_checkpoint == 0 and checkpoint_intersected == 17:
                        self.checkpoint_reward -= self.checkpoint_reward_step
                    elif checkpoint_intersected == self.last_checkpoint + 1:
                        self.checkpoint_reward += self.checkpoint_reward_step
                    elif checkpoint_intersected == self.last_checkpoint - 1:
                        self.checkpoint_reward -= self.checkpoint_reward_step
                    self.last_checkpoint = checkpoint_intersected
            else:
                pass
            if self.last_checkpoint == 17:
                center_reward = (500 - np.sqrt(np.sum((self.center_coord - env.checkpoint_center[0])**2)))*0.002
            else:
                center_reward = (500 - np.sqrt(np.sum((self.center_coord - env.checkpoint_center[self.last_checkpoint + 1]) ** 2)))*0.002
            self.total_reward = self.checkpoint_reward + center_reward
            return self.total_reward

    def draw_cars(self, car):
        self.frame = self.background.copy()
        # cv2.fillPoly(self.frame, [car.coordinates_int], (0, 0, 255))
        cv2.fillPoly(self.frame, np.array([car.coordinates_int], np.int), (0, 0, 255))

    def update_visualize(self, sensors_output, sensors_lines):
        if self.car_1.is_dead == 1:
            cv2.putText(self.frame, "Intersect detected, you are dead :(", (700, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(self.frame, "Show them who's the boss, racing boizzzz !!!!", (600, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
        for i in range(sensors_output.shape[0]):
            cv2.putText(self.frame, "Sensor %d: %.3f"
                        % (i + 1, sensors_output[i]), (30, 30 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 1, cv2.LINE_AA)
        if sensors_output.min() != -1:
            for i in range(len(sensors_lines)):
                cv2.line(self.frame, tuple(sensors_lines[i][0]), tuple(sensors_lines[i][1]),
                         (64, 64, 242), 1)

        for i in range(self.checkpoint_lines.shape[0]):
            cv2.line(self.frame, tuple(self.checkpoint_lines[i][0]), tuple(self.checkpoint_lines[i][1]),
                     (128, 0, 128), 2)
            cv2.circle(self.frame, tuple(self.checkpoint_center[i]), 2, (255, 255, 255), -1)
        cv2.putText(self.frame, "Reward is: %0.4f" % self.car_1.total_reward, (600, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
        frame_ldu = self.frame.swapaxes(0, 1)
        frame_ldu = cv2.cvtColor(frame_ldu, cv2.COLOR_BGR2RGB)
        my_surface = pygame.pixelcopy.make_surface(frame_ldu)
        self.display_surface.blit(my_surface, (0, 0))
        pygame.display.update()

    def get_keyboard_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        keys = pygame.key.get_pressed()
        speed = 0
        rotate_angle_speed = 0
        if keys[pygame.K_ESCAPE]:
            pygame.quit()
            quit()
        if keys[pygame.K_RIGHT]:
            rotate_angle_speed = np.pi / 30
        if keys[pygame.K_LEFT]:
            rotate_angle_speed = -np.pi / 30
        if keys[pygame.K_UP]:
            speed = 10
        if keys[pygame.K_DOWN]:
            speed = -10
        if keys[pygame.K_SPACE]:
            # reset_function()
            self.car_1.__init__(visualize_enable=self.visualize_enable)

        return speed, rotate_angle_speed

    def init_pygame(self):
        pygame.init()
        self.display_surface = pygame.display.set_mode((1000, 700))
        pygame.display.set_caption('Racingboiz simulator')


def visualize():
    env = CarTrackSimulator(visualize_enable=True)
    while True:
        velocity, rotate_angle_speed = env.get_keyboard_input()
        sensors_output, reward, terminate = env.step(1.5 * velocity, rotate_angle_speed)
        print(sensors_output, reward, terminate)


def non_visualize():
    env = CarTrackSimulator(visualize_enable=False)
    while True:
        sensors_output, reward, terminate = env.step(1, 2)
        print(sensors_output, reward, terminate)


if __name__ == "__main__":
    visualize()
