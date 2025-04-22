import pandas as pd
import os

# Путь к папке с датасетами
folder_path = r"C:\Users\funny\OneDrive\Desktop\4Course_Project_HSE\codex"

# Список для хранения всех DataFrame
all_dataframes = []

# Проход по всем файлам в папке
for file_name in os.listdir(folder_path):
    # Полный путь к файлу
    file_path = os.path.join(folder_path, file_name)
    
    # Загрузка файла в DataFrame
    if file_name.endswith(".xlsx"):  # Если файл в формате Excel
        df = pd.read_excel(file_path)
    elif file_name.endswith(".csv"):  # Если файл в формате CSV
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    else:
        continue  # Пропустить файлы других форматов
    
    # Добавление DataFrame в список
    all_dataframes.append(df)

# Объединение всех DataFrame в один
combined_df = pd.concat(all_dataframes, ignore_index=True)

# Сохранение объединенного датасета
output_xlsx_path = "объединенный_датасет.xlsx"
output_csv_path = "объединенный_датасет.csv"

# Сохранение в Excel
combined_df.to_excel(output_xlsx_path, index=False)

# Сохранение в CSV
combined_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

print(f"Объединенный датасет сохранен в файлы: {output_xlsx_path} и {output_csv_path}")