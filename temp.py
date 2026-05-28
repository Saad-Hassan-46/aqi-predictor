import tempfile

path = r"C:\Users\saadh\AppData\Local\Programs\Python\Python311\Lib\site-packages\hopsworks_common\client\base.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

fixed = content.replace(
    'os.path.join("/tmp"',
    'os.path.join(tempfile.gettempdir()'
)

with open(path, "w", encoding="utf-8") as f:
    f.write(fixed)

print("Done - patched", fixed.count("tempfile.gettempdir()"), "occurrences")