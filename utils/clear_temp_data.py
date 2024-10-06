import os
import shutil
from config import TEMP_DATA_PATH, UPLOADS_DIR

def clear_temp_and_uploads():
    """
    Clear the contents of the temporary data and uploads directories.
    """
    directories_to_clear = [TEMP_DATA_PATH, UPLOADS_DIR]

    for directory in directories_to_clear:
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        else:
            print(f"Directory not found: {directory}")

    print("Temporary data and uploads directories have been cleared.")

if __name__ == "__main__":
    clear_temp_and_uploads()
