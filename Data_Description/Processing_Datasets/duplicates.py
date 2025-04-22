import pandas as pd

# Пути к файлам
xlsx_file_path = r"C:\Users\funny\OneDrive\Desktop\4Course_Project_HSE\codex\объединенный_датасет.xlsx"
csv_file_path = r"C:\Users\funny\OneDrive\Desktop\4Course_Project_HSE\codex\объединенный_датасет.csv"

# Функция для проверки и удаления дублей
def remove_duplicates(df, subset_columns):

    # Поиск дубликатов
    duplicates = df.duplicated(subset=subset_columns, keep=False)
    
    # Вывод дубликатов (для проверки)
    print("Найдены дубликаты:")
    print(df[duplicates])
    
    # Удаление дубликатов
    cleaned_df = df.drop_duplicates(subset=subset_columns, keep="first")
    
    return cleaned_df, df[duplicates]

# Загрузка и обработка XLSX файла
df_xlsx = pd.read_excel(xlsx_file_path)
cleaned_xlsx, duplicates_xlsx = remove_duplicates(df_xlsx, subset_columns=["Вопрос", "Ответ"])

# Сохранение очищенного XLSX файла
cleaned_xlsx_path = r"C:\Users\funny\OneDrive\Desktop\4Course_Project_HSE\codex\очищенный_датасет.xlsx"
cleaned_xlsx.to_excel(cleaned_xlsx_path, index=False)

# Загрузка и обработка CSV файла
df_csv = pd.read_csv(csv_file_path, encoding="utf-8-sig")
cleaned_csv, duplicates_csv = remove_duplicates(df_csv, subset_columns=["Вопрос", "Ответ"])

# Сохранение очищенного CSV файла
cleaned_csv_path = r"C:\Users\funny\OneDrive\Desktop\4Course_Project_HSE\codex\очищенный_датасет.csv"
cleaned_csv.to_csv(cleaned_csv_path, index=False, encoding="utf-8-sig")

print(f"Очищенные данные сохранены в файлы: {cleaned_xlsx_path} и {cleaned_csv_path}")


