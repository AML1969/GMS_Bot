# Функция для замены номеров классов в метках
def replace_class_ids_in_labels(label_dir, matches_file):
    # Чтение соответствия классов из файла
    class_matches = {}
    counter_of_class = {} # счетчик классов
    with open(matches_file, 'r') as file:
        for line in file:
            parts = line.strip().split('|')
            if len(parts) == 2:
                old_class_id, new_class_id = map(int, parts)
                class_matches[old_class_id] = new_class_id
                counter_of_class[old_class_id] = 0

    print("Количество заменяемых классов = ",len(class_matches))

    
    # Обработка файлов меток
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    for label_file in tqdm(label_files, desc="Замена классов в метках", unit="file"):
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, 'r') as file:
            lines = file.readlines()

        with open(label_path, 'w') as file:
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 1:
                    class_id = int(parts[0])
                    if class_id in class_matches:
                        parts[0] = str(class_matches[class_id])
                        counter_of_class[class_id] += 1
                    file.write(' '.join(parts) + '\n')
    print("Статистика по классам : ",counter_of_class)