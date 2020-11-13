import numpy as np
import cv2
import pygame
import os
from shapely.geometry import LineString


class Environment:
    def __init__(self):
        self.width = 1000
        self.height = 700
        self.Race_track_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "Car_track_model", "Car_track_model/track_0.2.png")
        self.background = cv2.imread(self.Race_track_path, cv2.IMREAD_COLOR)
        self.frame = self.background.copy()
        self.checkpoint_lines = np.array([[(63, 307), (249, 306)], [(108, 126), (250, 271)], [(255, 79), (286, 244)],
                                          [(369, 84), (379, 252)], [(459, 88), (478, 261)], [(553, 85), (579, 288)],
                                          [(630, 79), (690, 285)], [(771, 64), (756, 286)], [(940, 117), (783, 285)],
                                          [(981, 301), (772, 339)], [(964, 489), (766, 382)], [(928, 603), (714, 441)],
                                          [(669, 630), (657, 442)], [(543, 414), (546, 610)], [(444, 391), (418, 588)],
                                          [(328, 370), (303, 577)], [(258, 361), (138, 574)], [(246, 343), (49, 447)]])

    def draw_cars(self, car_class):
        self.frame = self.background.copy()
        for car in car_class:
            cv2.fillPoly(self.frame, [car.coordinates_int], (0, 0, 255))


class IterCar(type):
    def __iter__(cls):
        return iter(cls.allCar)


class Car(metaclass=IterCar):
    allCar = []

    def __init__(self, visualize_enable):

        self.reward = 0
        self.visualize = visualize_enable
        self.width = np.float(20)
        self.height = np.float(10)
        self.diagonal = np.sqrt((self.width / 2) ** 2 + (self.height / 2) ** 2)
        self.alpha = np.arctan(self.height / self.width)
        self.sin_alpha = np.sin(self.alpha)
        self.cos_alpha = np.cos(self.alpha)
        self.center_coord = np.array([170, 350], np.float)
        self.rotate_angle = np.float(-np.pi / 2)
        self.coordinates_int = (self.diagonal * np.array(
            [[self.cos_alpha, self.sin_alpha],
             [self.cos_alpha, -self.sin_alpha],
             [-self.cos_alpha, -self.sin_alpha],
             [-self.cos_alpha, self.sin_alpha]],
            np.float) + np.repeat([self.center_coord], 4, axis=0) + 0.5).astype(np.uint)
        self.is_dead = 0
        self.last_checkpoint = 17

    def update_coords(self, rotate_angle_speed_ldf, speed_ldu):
        if self.is_dead == 0:
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
        # cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
        for x in np.arange(start_x, end_x, 1):
            for y in np.arange(start_y, end_y, 1):
                if (background[int(y), int(x)] == (0, 0, 0)).all() and \
                        (frame[int(y), int(x)] == (0, 0, 255)).all():
                    if self.is_dead == 0:
                        self.reward -= 5
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
                    self.reward += 1
                elif self.last_checkpoint == 0 and checkpoint_intersected == 17:
                    self.reward -= 1
                elif checkpoint_intersected == self.last_checkpoint + 1:
                    self.reward += 1
                elif checkpoint_intersected == self.last_checkpoint - 1:
                    self.reward -= 1
                self.last_checkpoint = checkpoint_intersected
        else:
            pass
        return self.reward

    def update_all(self, velocity, rotate_angle_speed, env):
        reward = self.get_reward(env)
        self.check_border_intersect(env.background, env.frame)
        self.update_coords(rotate_angle_speed, velocity)
        if self.visualize:
            sensor_1_output, sensor_1_line = self.get_sensor_output(env, self.rotate_angle)
            sensor_2_output, sensor_2_line = self.get_sensor_output(env, self.rotate_angle + np.pi / 4)
            sensor_3_output, sensor_3_line = self.get_sensor_output(env, self.rotate_angle - np.pi / 4)
            sensor_4_output, sensor_4_line = self.get_sensor_output(env, self.rotate_angle + np.pi * 3.2 / 4)
            sensor_5_output, sensor_5_line = self.get_sensor_output(env, self.rotate_angle - np.pi * 3.2 / 4)
            sensors_output = np.array([sensor_1_output, sensor_2_output,
                                       sensor_3_output, sensor_4_output, sensor_5_output])
            sensors_lines = np.array([sensor_1_line, sensor_2_line, sensor_3_line, sensor_4_line, sensor_5_line])
            return sensors_output, reward, self.is_dead, sensors_lines
        else:
            sensor_1_output = self.get_sensor_output(env, self.rotate_angle)
            sensor_2_output = self.get_sensor_output(env, self.rotate_angle + np.pi / 4)
            sensor_3_output = self.get_sensor_output(env, self.rotate_angle - np.pi / 4)
            sensor_4_output = self.get_sensor_output(env, self.rotate_angle + np.pi * 3.2 / 4)
            sensor_5_output = self.get_sensor_output(env, self.rotate_angle - np.pi * 3.2 / 4)
            sensors_output = np.array([sensor_1_output, sensor_2_output,
                                       sensor_3_output, sensor_4_output, sensor_5_output])
            return sensors_output, reward, self.is_dead


def init_simulator(graphic_enable):
    env = Environment()
    pygame.init()
    if graphic_enable:
        display_surface = pygame.display.set_mode((1000, 700))
        pygame.display.set_caption('Racingboiz simulator')
        return env, display_surface
    else:
        return env


def get_keyboard_input():
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
        clean_up_cars(Car)
        visualize()
    return speed, rotate_angle_speed


def update_visualize_window(display_surface, env, car, sensors_output, sensors_lines):
    if car.is_dead == 1:
        cv2.putText(env.frame, "Intersect detected, you are dead :(", (700, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 200), 1, cv2.LINE_AA)
    else:
        cv2.putText(env.frame, "Show them who's the boss, racing boizzzz !!!!", (600, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
    for i in range(sensors_output.shape[0]):
        cv2.putText(env.frame, "Sensor %d: %.3f"
                    % (i + 1, sensors_output[i]), (30, 30 + 30 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 1, cv2.LINE_AA)
    if sensors_output.min() != -1:
        for i in range(len(sensors_lines)):
            cv2.line(env.frame, tuple(sensors_lines[i][0]), tuple(sensors_lines[i][1]),
                     (64, 64, 242), 1)

    for i in range(env.checkpoint_lines.shape[0]):
        cv2.line(env.frame, tuple(env.checkpoint_lines[i][0]), tuple(env.checkpoint_lines[i][1]), (128, 0, 128), 2)
    cv2.putText(env.frame, "Reward is: %d" % car.checkpoint_reward, (600, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
    frame_ldu = env.frame.swapaxes(0, 1)
    frame_ldu = cv2.cvtColor(frame_ldu, cv2.COLOR_BGR2RGB)
    my_surface = pygame.pixelcopy.make_surface(frame_ldu)
    display_surface.blit(my_surface, (0, 0))
    pygame.display.update()
    pygame.time.delay(10)


def clean_up_cars(car_class):
    for car in car_class:
        if car.is_dead == 1:
            car_class.allCar.pop(car_class.allCar.index(car))


def visualize():
    env, display_surface = init_simulator(graphic_enable=True)
    car_1 = Car(visualize_enable=True)
    while True:
        sensors_lines = []
        velocity, rotate_angle_speed = get_keyboard_input()
        sensors_output, reward, terminate, sensors_lines[len(sensors_lines):] \
            = car_1.update_all(velocity, rotate_angle_speed, env)
        env.draw_cars(Car)
        update_visualize_window(display_surface, env, car_1, sensors_output, sensors_lines)


def non_visualize():
    env = init_simulator(graphic_enable=False)
    car_1 = Car(visualize_enable=False)
    while True:
        velocity = 0
        rotate_angle_speed = 0
        sensors_output, reward, terminate = car_1.update_all(velocity, rotate_angle_speed, env)
        env.draw_cars(Car)


if __name__ == "__main__":
    visualize()
