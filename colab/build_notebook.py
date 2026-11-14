"""Builds colab/XFND_CLIP_Server.ipynb from the cells below and colab/server.py.

    uv run python colab/build_notebook.py

Keeping the notebook generated means the server code inside it is always
exactly colab/server.py (which is what the tests check).
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


INTRO = """
# XFND-CLIP: model server for the website

This notebook runs your trained XFND-CLIP model on Colab's GPU and connects it to the website.

**Before starting:** `Runtime → Change runtime type → T4 GPU → Save`.

Run the boxes **from top to bottom**. Wait for the green tick ✅ before the next one.

| Box | What it does | First time | Demo day |
|---|---|---|---|
| 1 | Connects Google Drive and loads the project code | 2 min (upload `XFND-CLIP.zip`) | 1 min |
| 2 | Gets the trained model | 30–40 min (upload `archive.zip`, trains once, saves to Drive) | 1 min (loads from Drive) |
| 3 | Prepares the server | seconds | seconds |
| 4 | **Starts the model and prints the link for the website** | 2–3 min | 2–3 min |
| 5 | Stops the server | | |

Your files are kept in **Google Drive → `XFND`**, so you upload them only once.
"""

BOX1 = r"""
# ===== Box 1: connect Google Drive and load the project code =====
import os, glob, shutil, zipfile
from google.colab import drive, files

drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/XFND'
os.makedirs(DRIVE, exist_ok=True)

code_zip = os.path.join(DRIVE, 'XFND-CLIP.zip')
if not os.path.exists(code_zip):
    print('👉 Choose XFND-CLIP.zip (it is saved to Google Drive/XFND, so this happens only once)')
    uploaded = files.upload()
    shutil.move(next(iter(uploaded)), code_zip)

for d in ('/content/XFND-CLIP', '/content/tmp'):
    shutil.rmtree(d, ignore_errors=True)
zipfile.ZipFile(code_zip).extractall('/content/tmp')
shutil.move(os.path.dirname(glob.glob('/content/tmp/**/config.py', recursive=True)[0]), '/content/XFND-CLIP')

# Safety: load the saved model as data only (it can never run code).
import re
p = '/content/XFND-CLIP/explain.py'
s = open(p).read()
if 'weights_only' in s:
    print('Model loading is already safe (weights_only=True).')
else:
    s2 = re.sub(r'map_location=C\.DEVICE\)', 'map_location=C.DEVICE, weights_only=True)', s, count=1)
    open(p, 'w').write(s2)
    print('Model loading made safe (weights_only=True).' if s2 != s else '⚠️ Could not apply the safety patch.')

print('✅ Code ready - go to Box 2')
"""

BOX2 = r"""
# ===== Box 2: get the trained model =====
import os, glob, shutil, subprocess, zipfile
from google.colab import files
os.chdir('/content/XFND-CLIP')
subprocess.run('pip install -q -r requirements.txt', shell=True, check=True)
os.makedirs('checkpoints', exist_ok=True)
ck_drive = os.path.join(DRIVE, 'xfnd_clip.pt')

if os.path.exists(ck_drive):
    shutil.copy(ck_drive, 'checkpoints/xfnd_clip.pt')
    print('✅ Trained model loaded from Google Drive - go to Box 3')
else:
    print('No saved model yet: training once (30-40 min). Keep this tab open.')
    data_zip = os.path.join(DRIVE, 'archive.zip')
    if not os.path.exists(data_zip):
        print('👉 Choose archive.zip (the IFND dataset; saved to Google Drive/XFND)')
        uploaded = files.upload()
        shutil.move(next(iter(uploaded)), data_zip)
    shutil.rmtree('/content/ds', ignore_errors=True)
    zipfile.ZipFile(data_zip).extractall('/content/ds')
    csvs = glob.glob('/content/ds/**/*.csv', recursive=True)
    pick = [c for c in csvs if 'ifnd' in os.path.basename(c).lower()] or sorted(csvs, key=os.path.getsize)[-1:]
    os.makedirs('data', exist_ok=True)
    shutil.copy(pick[0], 'data/IFND.csv')
    for cmd in ['python data_prep.py', 'python features.py', 'python train.py']:
        print('\n>>> RUNNING:', cmd, flush=True)
        if subprocess.run(cmd, shell=True).returncode != 0:
            raise SystemExit(f'❌ Stopped at: {cmd} - send a screenshot of the error')
    shutil.copy('checkpoints/xfnd_clip.pt', ck_drive)
    if os.path.exists('results/metrics.json'):
        shutil.copy('results/metrics.json', os.path.join(DRIVE, 'metrics.json'))
    print('\n✅ Model trained and saved to Google Drive/XFND - next time Box 2 takes 1 minute. Go to Box 3')
"""

BOX3_MD = """
## Box 3: prepare the server
Just run it. It writes `server.py`, the small program that answers the website.
"""

BOX4 = r"""
# ===== Box 4: start the model and get the link for the website =====
# Your website address (the Render link). Change it once after deploying the site.
SITE_URL = 'https://xfnd-clip.onrender.com'

import os, re, secrets, subprocess, time, urllib.error, urllib.parse, urllib.request
from IPython.display import HTML, display
os.chdir('/content/XFND-CLIP')
subprocess.run('pip install -q fastapi "uvicorn[standard]" python-multipart', shell=True, check=True)
def stop_old():
    # No shell here: "pkill -f" inside a shell would also match (and kill) that shell.
    subprocess.run(['pkill', '-f', 'uvicorn server:app'])
    subprocess.run(['pkill', '-f', 'cloudflared tunnel'])
    import socket
    for _ in range(20):                      # wait until port 8000 is free again
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1', 8000)) != 0:
                return
        time.sleep(0.5)
stop_old()

KEY = secrets.token_urlsafe(24)   # new secret every run: only people with the link can use your GPU
# Only your website may call the server from a browser.
ORIGINS = ','.join([SITE_URL.rstrip('/'), 'http://127.0.0.1:8080', 'http://localhost:8080'])
server = subprocess.Popen(['python', '-m', 'uvicorn', 'server:app', '--host', '127.0.0.1', '--port', '8000'],
                          env=dict(os.environ, XFND_ACCESS_KEY=KEY, XFND_ALLOWED_ORIGINS=ORIGINS),
                          stdout=open('server.log', 'w'), stderr=subprocess.STDOUT)

print('Loading BERT, ResNet-50, CLIP and your model (about 1-2 minutes)...', flush=True)
deadline = time.time() + 600
while True:
    if server.poll() is not None:
        print(open('server.log').read()[-3000:])
        raise SystemExit('❌ The server stopped. Send a screenshot of the text above.')
    try:
        req = urllib.request.Request('http://127.0.0.1:8000/api/health', headers={'X-Access-Key': KEY})
        if urllib.request.urlopen(req, timeout=300).status == 200:
            break
    except urllib.error.HTTPError as err:
        if err.code == 500:                  # the model could not load: show why and stop
            print(err.read().decode(errors='replace'))
            print(open('server.log').read()[-3000:])
            raise SystemExit('❌ The model failed to load (see above). Did Box 2 finish with a green tick?')
        time.sleep(3)
    except Exception:
        if time.time() > deadline:
            raise SystemExit('❌ The model took too long to load. Run Box 4 again.')
        time.sleep(3)
print('✅ Model loaded')

# Cloudflare's tunnel program, a fixed version checked against its published SHA-256.
CF_URL = 'https://github.com/cloudflare/cloudflared/releases/download/2026.11.0/cloudflared-linux-amd64'
CF_SHA256 = '51f7c68b57ed7269164075d7e789efce5b59355982f06392deeb037d982a51fd'
import hashlib
if not os.path.exists('cloudflared') or hashlib.sha256(open('cloudflared', 'rb').read()).hexdigest() != CF_SHA256:
    urllib.request.urlretrieve(CF_URL, 'cloudflared')
    if hashlib.sha256(open('cloudflared', 'rb').read()).hexdigest() != CF_SHA256:
        os.remove('cloudflared')
        raise SystemExit('❌ The tunnel program failed its checksum. Run Box 4 again.')
    os.chmod('cloudflared', 0o755)
tunnel = subprocess.Popen(['./cloudflared', 'tunnel', '--url', 'http://127.0.0.1:8000', '--no-autoupdate'],
                          stdout=open('tunnel.log', 'w'), stderr=subprocess.STDOUT)
url = None
for _ in range(90):
    m = re.search(r'https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com', open('tunnel.log').read())
    if m:
        url = m.group(0)
        break
    time.sleep(1)
if not url:
    raise SystemExit('❌ Could not create the public link. Run Box 4 again.')

CODE = f'{url}#key={KEY}'
LINK = f"{SITE_URL.rstrip('/')}/#connect={urllib.parse.quote(CODE, safe='')}"
display(HTML(f'''
<div style="font:15px/1.5 sans-serif;padding:16px;border:2px solid #0f9d58;border-radius:10px">
  <b style="font-size:18px">✅ The model is online</b><br><br>
  <b>Easiest:</b> <a href="{LINK}" target="_blank" style="font-size:16px">open the website</a> and press <b>Connect</b><br><br>
  <b>Or</b> copy this connection code into the website's <i>Connect model</i> box:<br>
  <code style="user-select:all;word-break:break-all">{CODE}</code><br><br>
  Works only from <b>{SITE_URL}</b>. Keep this Colab tab open during the demo. Stop with Box 5.<br>
  <small>Before sharing this notebook with anyone: <i>Edit → Clear all outputs</i> (this box contains the secret key).</small>
</div>'''))
"""

BOX5 = r"""
# ===== Box 5: stop the server =====
import subprocess
subprocess.run(['pkill', '-f', 'uvicorn server:app'])
subprocess.run(['pkill', '-f', 'cloudflared tunnel'])
print('Stopped. The website now shows the model as offline.')
"""


def build():
    server_src = (HERE / "server.py").read_text()
    cells = [
        md(INTRO),
        md("## Box 1: connect Google Drive and load the project code\n"
           "The first time, a **Choose Files** button appears: pick `XFND-CLIP.zip`."),
        code(BOX1),
        md("## Box 2: get the trained model\n"
           "**First time:** pick `archive.zip` when asked. It trains once (30–40 min) and saves the model "
           "to Google Drive.\n\n**Demo day:** loads the saved model in about a minute."),
        code(BOX2),
        md(BOX3_MD),
        code("%%writefile /content/XFND-CLIP/server.py\n" + server_src),
        md("## Box 4: start the model and get the link\n"
           "Wait for the green box, click **open the website**, then press **Connect**.\n\n"
           "First time only: after deploying the website, put its address in `SITE_URL` below."),
        code(BOX4),
        md("## Box 5: stop the server (after the demo)"),
        code(BOX5),
    ]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 0}
    out = HERE / "XFND_CLIP_Server.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    build()
