path = r"C:\Users\saadh\AppData\Local\Programs\Python\Python311\Lib\site-packages\hopsworks_common\client\base.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

if "import tempfile" not in content:
    content = "import tempfile\n" + content
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Added tempfile import")
else:
    print("Already imported")