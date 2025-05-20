import os


def count_files(directory = '.', recursive=True):
    """
    Подсчитывает количество файлов в указанной директории.

    :param directory: Путь к директории
    :param recursive: Если True, считает файлы в поддиректориях
    :return: Количество файлов
    """
    file_count = 0

    if recursive:
        for root, dirs, files in os.walk(directory):
            file_count += len(files)
    else:
        file_count = len([name for name in os.listdir(directory)
                          if os.path.isfile(os.path.join(directory, name))])

    return file_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Подсчет файлов в директории')
    parser.add_argument('directory', nargs='?', default='./process_names/processes_logs', help='Путь к директории')
    parser.add_argument('--no-recursive', dest='recursive', action='store_false',
                        help='Не считать файлы в поддиректориях')
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Ошибка: '{args.directory}' не является директорией или не существует")
        exit(1)

    count = count_files(args.directory, args.recursive)
    print(f"Найдено файлов: {count}")