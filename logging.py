from datetime import datetime


class Logger:
    """Handles detailed logging for training progress."""
    def __init__(self, log_path='training_log.txt'):
        self.log_path = log_path

    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_message = f'[{timestamp}] {message}'
        print(formatted_message)
        with open(self.log_path, 'a') as f:
            f.write(formatted_message + '\n')
