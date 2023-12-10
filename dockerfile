# Используем базовый образ PyTorch
FROM pytorch/pytorch:latest

# Устанавливаем необходимые библиотеки
RUN pip install pandas
RUN pip install scikit-learn

# Копируем файлы проекта в контейнер
COPY ./train.py /app/train.py
COPY ./trainAgain.py /app/trainAgain.py
COPY ./model.pth /app/model.pth
COPY ./test.py /app/start.py

# Задаем рабочую директорию
WORKDIR /app

# Команда для запуска по умолчанию
CMD ["python", "test.py"]
