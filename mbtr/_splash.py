import os.path


def splash():
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'splash.txt'), 'r') as splash_file:
        print(splash_file.read())
