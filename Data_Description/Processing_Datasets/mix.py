import pandas as pd

# Пути к файлам
xlsx_file_path = r"C:\Users\funny\OneDrive\Desktop\4Course_Project_HSE\codex\очищенный_датасет.xlsx"
csv_file_path = r"C:\Users\funny\OneDrive\Desktop\4Course_Project_HSE\codex\очищенный_датасет.csv"

# Загрузка данных
df_xlsx = pd.read_excel(xlsx_file_path)
df_csv = pd.read_csv(csv_file_path, encoding="utf-8-sig")

# Перемешивание данных
shuffled_xlsx = df_xlsx.sample(frac=1, random_state=42).reset_index(drop=True)
shuffled_csv = df_csv.sample(frac=1, random_state=42).reset_index(drop=True)

# Сохранение перемешанных данных
shuffled_xlsx_path = r"C:\Users\funny\OneDrive\Desktop\4Course_Project_HSE\codex\перемешанный_датасет.xlsx"
shuffled_csv_path = r"C:\Users\funny\OneDrive\Desktop\4Course_Project_HSE\codex\перемешанный_датасет.csv"

shuffled_xlsx.to_excel(shuffled_xlsx_path, index=False)
shuffled_csv.to_csv(shuffled_csv_path, index=False, encoding="utf-8-sig")

print(f"Перемешанные данные сохранены в файлы: {shuffled_xlsx_path} и {shuffled_csv_path}")