import os
import shutil


class PathUtils:
    @staticmethod
    def create_dir(folder_path, show_message=True):
        if not os.path.exists(folder_path):
            if show_message:
                print("Creating folder: {}".format(folder_path))
            os.makedirs(folder_path)

    @staticmethod
    def remove_dir(folder_path):
        if os.path.exists(folder_path):
            print("Removing folder: {}".format(folder_path))
            shutil.rmtree(folder_path)
