# שימוש בתמונת פייתון קלה
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    graphviz \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

# התקנת התלויות - הפעם מורידים את הגרסה המלאה מהחנות הרגילה שכוללת תמיכה ב-GPU
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# פקודת ההפעלה של שרת המחברות
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]