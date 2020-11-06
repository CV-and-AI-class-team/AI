import numpy as np
import cv2
import pygame
import os


class Environment:
    def __init__(self):
        self.width = 1000
        self.height = 700
        self.Race_track_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "Car_track_model", "track_0.2.png")
        self.background = cv2.imread(self.Race_track_path, cv2.IMREAD_COLOR)
        self.frame = self.background.copy()

    def update_env(self, car_coord):
        self.frame = self.background.copy()
        cv2.fillPoly(self.frame, [car_coord], (0, 0, 255))


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
        self.is_dead = 0

    def update_coords(self, rotate_angle_speed_ldf, speed_ldu):
        if self.is_dead == 0:
            # if speed_ldu > 0:
            self.center_coord += np.array([speed_ldu * np.cos(self.rotate_angle),
                                           speed_ldu * np.sin(self.rotate_angle)],
                                          np.float)
            self.center_coord[0] = max((min(self.center_coord[0], np.uint(995-max(self.width, self.height)/2))),
                                       np.uint(max(self.width, self.height)/2+5))
            self.center_coord[1] = max((min(self.center_coord[1], np.uint(695-max(self.width, self.height)/2))),
                                       np.uint(max(self.width, self.height)/2+5))

            if self.rotate_angle >= 2*np.pi:
                self.rotate_angle -= 2*np.pi
            elif self.rotate_angle <= -2*np.pi:
                self.rotate_angle += 2*np.pi

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
        else:
            pass

    def check_border_intersect(self, background, frame):
        start_x = np.min(self.coordinates_int[:, 0])
        start_y = np.min(self.coordinates_int[:, 1])
        end_x = np.max(self.coordinates_int[:, 0])
        end_y = np.max(self.coordinates_int[:, 1])
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
        for x in np.arange(start_x, end_x, 1):
            for y in np.arange(start_y, end_y, 1):
                if (background[int(y), int(x)] == (0, 0, 0)).all() and \
                        (frame[int(y), int(x)] == (0, 0, 255)).all():
                    self.is_dead = 1
                    return 1
        return 0

    def get_sensor_output(self, background, sensor_angle):
        line_array = self.get_sensor_line(background.shape, sensor_angle)
        distance = -1
        sensor_range = np.array([[line_array[0, 0], line_array[1, 0]], [line_array[0, -1], line_array[1, -1]]])
        for index in range(line_array.shape[1]):
            if (background[line_array[1, index], line_array[0, index]] == (0, 0, 0)).all():
                distance = np.sqrt((self.center_coord[1] +
                                    np.sin(self.rotate_angle)*self.diagonal-line_array[1, index]) ** 2 +
                                   (self.center_coord[0] +
                                    np.cos(self.rotate_angle)*self.diagonal-line_array[0, index]) ** 2)
                sensor_range[1][0] = line_array[0, index]
                sensor_range[1][1] = line_array[1, index]
                break
        cv2.line(env.frame, tuple(sensor_range[0]), tuple(sensor_range[1]),
                 (0, 0, 255), 2)
        return distance

    def get_sensor_line(self, frame_size, angle):
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        if cos_angle == 0:
            t = np.arange(0, (frame_size[0]-5-self.center_coord[1])/sin_angle, 1/sin_angle)
        elif sin_angle == 0:
            t = np.arange(0, (frame_size[1]-5-self.center_coord[0])/cos_angle, 1/cos_angle)
        elif sin_angle > 0:
            if cos_angle > 0:
                t = np.arange(0, min((frame_size[1]-5-self.center_coord[0])/cos_angle,
                                     (frame_size[0]-5-self.center_coord[1])/sin_angle),
                              min(1/cos_angle, 1/sin_angle))
            else:
                t = np.arange(0, min(-(self.center_coord[0] - 5)/cos_angle,
                                     (frame_size[0]-5-self.center_coord[1])/sin_angle),
                              min(abs(1/cos_angle), abs(1/sin_angle)))
        else:
            if cos_angle > 0:
                t = np.arange(0, min(-(self.center_coord[1] - 5)/sin_angle,
                                     (frame_size[1]-5-self.center_coord[0])/cos_angle),
                              min(abs(1/cos_angle), abs(1/sin_angle)))
            else:
                t = np.arange(0, min(-(self.center_coord[0] - 5)/cos_angle,
                                     -(self.center_coord[1] - 5)/sin_angle),
                              min(abs(1/cos_angle), abs(1/sin_angle)))
        line_array = np.array([self.center_coord[0]+t*cos_angle, self.center_coord[1]+t*sin_angle], np.uint)
        return line_array


def visualize(frame_ldu, visualize_show_ldb, ):
    if visualize_show_ldb:
        frame_ldu = frame_ldu.swapaxes(0, 1)
        frame_ldu = cv2.cvtColor(frame_ldu, cv2.COLOR_BGR2RGB)
        my_surface = pygame.pixelcopy.make_surface(frame_ldu)
        display_surface.blit(my_surface, (0, 0))
        pygame.display.update()
    else:
        pass


def main():
    car_1 = Car(1)
    pygame.init()
    while True:
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
            rotate_angle_speed = np.pi/30
        if keys[pygame.K_LEFT]:
            rotate_angle_speed = -np.pi/30
        if keys[pygame.K_UP]:
            speed = 10
        if keys[pygame.K_DOWN]:
            speed = -10
        if keys[pygame.K_SPACE]:
            main()
        car_1.update_coords(rotate_angle_speed, speed)
        env.update_env(car_1.coordinates_int)
        sensor_1_output = car_1.get_sensor_output(env.background, car_1.rotate_angle - np.pi / 4)
        sensor_2_output = car_1.get_sensor_output(env.background, car_1.rotate_angle)
        sensor_3_output = car_1.get_sensor_output(env.background, car_1.rotate_angle + np.pi / 4)
        intersect_detect = car_1.check_border_intersect(env.background, env.frame)
        if intersect_detect == 1:
            cv2.putText(env.frame, "Intersect detected, you are dead :(", (700, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(env.frame, "Show them who's the boss, racing boizzzz !!!!", (600, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
        cv2.putText(env.frame, "Sensor 1: %.3f"
                    % sensor_1_output, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 1, cv2.LINE_AA)
        cv2.putText(env.frame, "Sensor 2: %.3f"
                    % sensor_2_output, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 1, cv2.LINE_AA)
        cv2.putText(env.frame, "Sensor 3: %.3f"
                    % sensor_3_output, (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 1, cv2.LINE_AA)
        visualize(env.frame, visualize_show)
        pygame.time.delay(10)


if __name__ == "__main__":
    env = Environment()
    pygame.init()
    visualize_show = True
    if visualize_show:
        display_surface = pygame.display.set_mode((1000, 700))
        pygame.display.set_caption('Racingboiz simulator')
    main()
