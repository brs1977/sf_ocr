import os

try:
    import fcntl
    def lock_file(file):
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)

    def unlock_file(file):
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)

except ImportError:
    import msvcrt

    def lock_file(file):
        # Блокируем весь файл (длина - размер файла)
        file_size = os.path.getsize(file.name) or 1
        msvcrt.locking(file.fileno(), msvcrt.LK_LOCK, file_size)

    def unlock_file(file):
        file_size = os.path.getsize(file.name) or 1
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, file_size)


