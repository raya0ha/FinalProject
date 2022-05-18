# This is a sample Python script.
from preprocessors import load_data, extract_feature
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    # load RAVDESS dataset, 75% training 25% testing
    X_train, X_test, y_train, y_test = load_data(test_size=0.25)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
