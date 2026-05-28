path = r"C:\Users\saadh\AppData\Local\Programs\Python\Python311\Lib\site-packages\hopsworks_common\constants.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

fixed = content.replace("class CLIENT:", "import tempfile\n\nclass CLIENT:", 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(fixed)

print("Done")