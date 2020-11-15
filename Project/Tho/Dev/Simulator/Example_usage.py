from simulator_lib_v1_1 import CarTrackSimulator


def main():
    environment = CarTrackSimulator(visualize_enable=True)
    while True:
        velocity, rotate_angle_speed = environment.get_keyboard_input()
        sensors_output, reward, terminate = environment.step(1.5 * velocity, rotate_angle_speed)
        print(sensors_output, reward, terminate)


if __name__ == '__main__':
    main()
