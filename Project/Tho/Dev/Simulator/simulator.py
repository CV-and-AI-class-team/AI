import numpy as np
import cv2
import pygame
import time


class Car:
    def __init__(self, car_num):
        self.car_num = car_num
        self.width = np.float(20)
        self.height = np.float(10)
        self.diagonal = np.sqrt((self.width/2)**2 + (self.height/2)**2)
        self.alpha = np.arctan(self.height/self.width)
        self.sin_alpha = np.sin(self.alpha)
        self.cos_alpha = np.cos(self.alpha)
        self.center_coord = np.array([200, 200], np.float)
        self.rotate_angle = np.float(0)
        self.coordinates_int = (self.diagonal * np.array(
            [[self.cos_alpha, self.sin_alpha],
             [self.cos_alpha, -self.sin_alpha],
             [-self.cos_alpha, -self.sin_alpha],
             [-self.cos_alpha, self.sin_alpha]],
            np.float) + np.repeat([self.center_coord], 4, axis=0) + 0.5).astype(np.uint)
        self.intersect = 0

    def calc_coords(self, rotate_angle_speed_ldf, speed_ldu):
        if speed_ldu > 0:
            self.center_coord += np.array([speed_ldu * np.cos(self.rotate_angle),
                                           speed_ldu * np.sin(self.rotate_angle)],
                                          np.float)
            if self.center_coord[0] - self.width/2 <= 5:
                self.center_coord[0] = np.uint(self.width/2 + 5)
            elif self.center_coord[0] + self.width/2 >= 995:
                self.center_coord[0] = np.uint(995 - self.width/2)

            if self.center_coord[1] - self.height/2 <= 5:
                self.center_coord[1] = np.uint(self.width/2 + 5)
            elif self.center_coord[1] + self.height/2 >= 695:
                self.center_coord[1] = np.uint(695 - self.width/2)

        if self.rotate_angle >= 2*np.pi:
            self.rotate_angle -= 2*np.pi
        elif self.rotate_angle <= -2 * np.pi:
            self.rotate_angle += 2 * np.pi

        self.rotate_angle += rotate_angle_speed_ldf
        sin_rotate = np.sin(self.rotate_angle)
        cos_rotate = np.cos(self.rotate_angle)
        cos_alpha_plus_rotate = self.cos_alpha*cos_rotate - self.sin_alpha*sin_rotate
        sin_alpha_plus_rotate = self.sin_alpha*cos_rotate + self.cos_alpha*sin_rotate
        cos_alpha_minus_rotate = self.cos_alpha*cos_rotate + self.sin_alpha*sin_rotate
        sin_alpha_minus_rotate = self.sin_alpha*cos_rotate - self.cos_alpha*sin_rotate

        self.coordinates_int = (self.diagonal * np.array(
            [[cos_alpha_plus_rotate, sin_alpha_plus_rotate],
             [cos_alpha_minus_rotate, -sin_alpha_minus_rotate],
             [-cos_alpha_plus_rotate, -sin_alpha_plus_rotate],
             [-cos_alpha_minus_rotate, sin_alpha_minus_rotate]],
            np.float) + np.repeat([self.center_coord], 4, axis=0) + 0.5).astype(np.uint)


class Environment:
    def __init__(self):
        self.width = 1000
        self.height = 700
        self.Race_track_path = "./Car_track_model/track_0.1.png"
        self.background = cv2.imread(self.Race_track_path, cv2.IMREAD_COLOR)
        self.frame = self.background.copy()

    def update_env(self, car_coord):
        self.frame = self.background.copy()
        cv2.fillPoly(self.frame, [car_coord], (255, 0, 0))

    def check_intersect(self, car_coord):
        start_x = np.min(car_coord[:, 0])
        start_y = np.min(car_coord[:, 1])
        end_x = np.max(car_coord[:, 0])
        end_y = np.max(car_coord[:, 1])
        cv2.rectangle(self.frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
        for x in np.arange(start_x, end_x, 1):
            for y in np.arange(start_y, end_y, 1):
                if (self.background[int(y), int(x)] == (0, 0, 0)).all() and \
                        (self.frame[int(y), int(x)] == (255, 0, 0)).all():
                    cv2.putText(self.frame, "Intersect detected", (300, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                                1, cv2.LINE_AA)
                    return ["Intersect detected"]
        cv2.putText(self.frame, "No Intersect", (300, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    1, cv2.LINE_AA)
        return ['No Intersect']


class KeyboardCode:
    left = 81
    right = 83
    up = 82
    down = 84
    space = 32


if __name__ == "__main__":
    env_1 = Environment()
    car_1 = Car(1)

    pygame.init()
    display_surface = pygame.display.set_mode((1000, 700))
    pygame.display.set_caption('Racing boiz')

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        keys = pygame.key.get_pressed()
        speed = 0
        rotate_angle_speed = 0
        if keys[pygame.K_RIGHT] and keys[pygame.K_LEFT]:
            rotate_angle_speed = 0
        elif keys[pygame.K_RIGHT]:
            rotate_angle_speed = np.pi/30
        elif keys[pygame.K_LEFT]:
            rotate_angle_speed = -np.pi/30
        if keys[pygame.K_SPACE]:
            speed = 10
        car_1.calc_coords(rotate_angle_speed, speed)
        env_1.update_env(car_1.coordinates_int)
        intersect_detect = env_1.check_intersect(car_1.coordinates_int)
        frame = env_1.frame.swapaxes(0, 1)

        my_surface = pygame.pixelcopy.make_surface(frame)
        display_surface.blit(my_surface, (0, 0))
        pygame.display.update()
        pygame.time.delay(10)
        # time.sleep(10)
