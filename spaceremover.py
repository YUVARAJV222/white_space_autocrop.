import os, io, base64, json, traceback, zipfile
from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageOps, ImageEnhance
import pytesseract
import numpy as np
import fitz  # pymupdf

app = Flask(__name__)
_results_store = {}  # token -> list of PIL images

# ── helpers ────────────────────────────────────────────────────────────────

def auto_crop_whitespace(img, border_px=0):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    arr = np.array(img.convert('L'))
    mask = arr < 245
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    h, w = arr.shape
    rmin = max(0, rmin - border_px)
    rmax = min(h - 1, rmax + border_px)
    cmin = max(0, cmin - border_px)
    cmax = min(w - 1, cmax + border_px)
    return img.crop((cmin, rmin, cmax + 1, rmax + 1))

def apply_effect(img, effect):
    if effect == 'grayscale':
        return img.convert('L').convert('RGB')
    elif effect == 'enhanced':
        img = ImageEnhance.Contrast(img).enhance(1.5)
        img = ImageEnhance.Sharpness(img).enhance(2.0)
        img = ImageEnhance.Brightness(img).enhance(1.1)
        return img
    elif effect == 'threshold':
        gray = img.convert('L')
        return gray.point(lambda x: 0 if x < 128 else 255, '1').convert('RGB')
    return img

def img_to_base64(img, fmt='PNG'):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

def pdf_page_to_pil(doc, page_idx, dpi=150):
    page = doc[page_idx]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def process_image(img, auto_crop, border_px, effect):
    orig = img.size
    if auto_crop:
        img = auto_crop_whitespace(img, border_px)
    elif border_px > 0:
        img = ImageOps.expand(img, border=border_px, fill='white')
    img = apply_effect(img, effect)
    return img, orig

# ── HTML ───────────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Image / PDF Processor</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',sans-serif;background:#1a1a2e;color:#eee;min-height:100vh}
header{background:#16213e;padding:16px 28px;display:flex;align-items:center;gap:12px;border-bottom:2px solid #0f3460}
header h1{font-size:1.35rem;color:#e94560;letter-spacing:1px}
header span{font-size:.8rem;color:#666}
.container{display:grid;grid-template-columns:310px 1fr;min-height:calc(100vh - 58px)}
.sidebar{background:#16213e;padding:18px;border-right:1px solid #0f3460;display:flex;flex-direction:column;gap:16px;overflow-y:auto}
.sec{font-size:.68rem;text-transform:uppercase;letter-spacing:2px;color:#555;margin-bottom:4px}
.drop-zone{border:2px dashed #0f3460;border-radius:10px;padding:20px 12px;text-align:center;cursor:pointer;transition:.2s}
.drop-zone:hover,.drop-zone.over{border-color:#e94560;background:rgba(233,69,96,.07)}
.drop-zone .ico{font-size:1.8rem;margin-bottom:6px}
.drop-zone p{font-size:.75rem;color:#666}
.drop-zone strong{color:#ccc;font-size:.85rem}
.opt-group{display:flex;flex-direction:column;gap:8px}
label{font-size:.8rem;color:#aaa}
input[type=range]{width:100%;accent-color:#e94560}
.rrow{display:flex;justify-content:space-between;align-items:center}
.egrid{display:grid;grid-template-columns:1fr 1fr;gap:7px}
.ebtn{background:#0f3460;border:2px solid transparent;border-radius:7px;padding:7px 4px;text-align:center;cursor:pointer;font-size:.75rem;color:#aaa;transition:.2s}
.ebtn:hover{border-color:#e94560;color:#fff}
.ebtn.on{border-color:#e94560;background:rgba(233,69,96,.15);color:#fff}
.trow{display:flex;justify-content:space-between;align-items:center}
.tog{position:relative;width:38px;height:21px}
.tog input{opacity:0;width:0;height:0}
.sl{position:absolute;inset:0;background:#0f3460;border-radius:21px;cursor:pointer;transition:.3s}
.sl:before{content:'';position:absolute;width:15px;height:15px;left:3px;top:3px;background:#fff;border-radius:50%;transition:.3s}
input:checked+.sl{background:#e94560}
input:checked+.sl:before{transform:translateX(17px)}
.btn{width:100%;padding:11px;border:none;border-radius:8px;font-size:.87rem;font-weight:600;cursor:pointer;transition:.2s;letter-spacing:.4px}
.bp{background:#e94560;color:#fff}
.bp:hover{background:#c73652;transform:translateY(-1px)}
.bp:disabled{background:#444;cursor:not-allowed;transform:none}
.prog-wrap{display:none;flex-direction:column;gap:6px}
.prog-wrap.show{display:flex}
.prog-bar-bg{background:#0d1b2a;border-radius:20px;height:8px;overflow:hidden}
.prog-bar{height:100%;background:linear-gradient(90deg,#e94560,#ff8c42);border-radius:20px;transition:width .4s;width:0%}
.prog-label{font-size:.75rem;color:#888;text-align:center}
.dl-panel{display:none;flex-direction:column;gap:8px}
.dl-panel.show{display:flex}
.dl-btn{width:100%;padding:11px 14px;border:none;border-radius:8px;font-size:.82rem;font-weight:600;cursor:pointer;display:flex;align-items:center;gap:8px;transition:.2s;text-decoration:none;justify-content:center}
.dl-pdf{background:#1a472a;color:#6fcf97;border:none}
.dl-pdf:hover{background:#246b3a}
.dl-zip{background:#1a2a47;color:#56b4f7;border:none}
.dl-zip:hover{background:#1f3a6e}
.dl-strip{background:#3d1a47;color:#c77dff;border:none}
.dl-strip:hover{background:#561e6b}
.dl-info{font-size:.72rem;color:#555;text-align:center;padding-top:2px}
.main{padding:20px;display:flex;flex-direction:column;gap:16px;overflow-y:auto}
.status-bar{background:#0d1b2a;border-radius:8px;padding:8px 14px;font-size:.77rem;color:#888;border-left:3px solid #e94560}
.preview-wrap{background:#16213e;border-radius:12px;padding:16px;border:1px solid #0f3460;min-height:260px;display:flex;flex-direction:column;gap:12px}
.preview-strip{display:flex;gap:8px;flex-wrap:wrap;align-items:flex-start;padding-bottom:4px;overflow-x:auto}
.page-card{display:flex;flex-direction:column;align-items:center;gap:4px;cursor:pointer;flex-shrink:0}
.page-card img{border:2px solid #0f3460;border-radius:5px;transition:.2s;max-height:150px;width:auto}
.page-card img:hover,.page-card img.sel{border-color:#e94560;box-shadow:0 0 8px rgba(233,69,96,.4)}
.page-card span{font-size:.68rem;color:#666}
.big-preview{display:flex;align-items:center;justify-content:center;min-height:200px}
.big-preview img{max-width:100%;max-height:60vh;border-radius:8px;box-shadow:0 4px 20px rgba(0,0,0,.5)}
.placeholder{text-align:center;color:#333;padding:40px;width:100%}
.placeholder .ico{font-size:3.5rem;margin-bottom:10px}
.results-row{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.card{background:#16213e;border-radius:10px;padding:14px;border:1px solid #0f3460}
.card h3{font-size:.75rem;text-transform:uppercase;letter-spacing:1.5px;color:#e94560;margin-bottom:8px}
.igrid{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.iitem{background:#0d1b2a;border-radius:5px;padding:7px 10px}
.iitem .k{font-size:.68rem;color:#555}
.iitem .v{font-size:.83rem;color:#ddd;font-weight:500}
.card pre{font-size:.75rem;color:#ccc;white-space:pre-wrap;word-break:break-word;max-height:180px;overflow-y:auto;background:#0d1b2a;padding:9px;border-radius:5px}
.copy-btn{float:right;background:#0f3460;border:none;color:#888;padding:3px 9px;border-radius:4px;cursor:pointer;font-size:.7rem}
.copy-btn:hover{background:#1a4a8a;color:#fff}
.overlay{display:none;position:fixed;inset:0;background:rgba(10,10,20,.8);z-index:999;align-items:center;justify-content:center;flex-direction:column;gap:14px}
.overlay.show{display:flex}
.spinner{width:46px;height:46px;border:3px solid #0f3460;border-top-color:#e94560;border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.omsg{font-size:.85rem;color:#aaa}
</style>
</head>
<body>

<div class="overlay" id="overlay">
  <div class="spinner"></div>
  <div class="omsg" id="omsg">Processing pages…</div>
</div>

<header>
  <h1>🖼️ Image &amp; PDF Processor</h1>
  <span>Auto-crop • Effects • OCR • All Pages → PDF / ZIP / Strip</span>
</header>

<div class="container">
<!-- SIDEBAR -->
<div class="sidebar">
  <div>
    <div class="sec">Upload</div>
    <div class="drop-zone" id="dz" onclick="document.getElementById('fi').click()">
      <div class="ico">📂</div>
      <strong>Click or Drop File</strong>
      <p>PNG · JPG · BMP · TIFF · PDF</p>
    </div>
    <input type="file" id="fi" accept="image/*,.pdf" style="display:none">
  </div>

  <div class="opt-group">
    <div class="sec">Crop Options</div>
    <div class="trow">
      <label>Auto-crop white borders</label>
      <label class="tog"><input type="checkbox" id="autoCrop" checked><span class="sl"></span></label>
    </div>
    <div style="margin-top:6px">
      <div class="rrow">
        <label>Add border (px)</label>
        <span id="bval" style="font-size:.83rem;color:#e94560;font-weight:700">0</span>
      </div>
      <input type="range" id="bpx" min="0" max="100" value="0"
             oninput="document.getElementById('bval').textContent=this.value">
    </div>
  </div>

  <div class="opt-group">
    <div class="sec">Visual Effect</div>
    <div class="egrid">
      <div class="ebtn on"  data-e="none"       onclick="selFx(this)">Original</div>
      <div class="ebtn"     data-e="grayscale"  onclick="selFx(this)">Grayscale</div>
      <div class="ebtn"     data-e="enhanced"   onclick="selFx(this)">Enhanced</div>
      <div class="ebtn"     data-e="threshold"  onclick="selFx(this)">B&amp;W Threshold</div>
    </div>
  </div>

  <div class="opt-group">
    <div class="sec">Extra Tools</div>
    <div class="trow">
      <label>Run OCR on all pages</label>
      <label class="tog"><input type="checkbox" id="doOCR"><span class="sl"></span></label>
    </div>
  </div>

  <button class="btn bp" id="procBtn" onclick="processAll()" disabled>⚡ Process All Pages</button>

  <div class="prog-wrap" id="progWrap">
    <div class="prog-bar-bg"><div class="prog-bar" id="progBar"></div></div>
    <div class="prog-label" id="progLabel">Processing…</div>
  </div>

  <div class="dl-panel" id="dlPanel">
    <div class="sec" style="margin-top:4px">⬇ Download Processed Output</div>
    <button class="dl-btn dl-pdf"   onclick="dlFile('pdf')">  📄 Download as PDF</button>
    <button class="dl-btn dl-zip"   onclick="dlFile('zip')">  🗜️ Download ZIP (all PNGs)</button>
    <button class="dl-btn dl-strip" onclick="dlFile('strip')">🖼️ Download Long PNG Strip</button>
    <div class="dl-info" id="dlInfo"></div>
  </div>
</div>

<!-- MAIN -->
<div class="main">
  <div id="statusBar" class="status-bar">Load an image or PDF to get started.</div>

  <div class="preview-wrap">
    <div class="placeholder" id="ph"><div class="ico">🖼️</div><p>Processed pages will appear here</p></div>
    <div class="preview-strip" id="thumbStrip" style="display:none"></div>
    <div class="big-preview"   id="bigPreview" style="display:none">
      <img id="bigImg" alt="Preview">
    </div>
  </div>

  <div class="results-row" id="infoRow" style="display:none">
    <div class="card">
      <h3>📐 Processing Info</h3>
      <div class="igrid" id="infoGrid"></div>
    </div>
    <div class="card" id="ocrCard" style="display:none">
      <h3>🔍 OCR Text <button class="copy-btn" onclick="copyOCR()">Copy</button></h3>
      <pre id="ocrPre"></pre>
    </div>
  </div>
</div>
</div>

<script>
let uploadedFile=null, fx='none', token=null, pageB64s=[];

const dz=document.getElementById('dz');
dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('over')});
dz.addEventListener('dragleave',()=>dz.classList.remove('over'));
dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('over');loadFile(e.dataTransfer.files[0])});
document.getElementById('fi').addEventListener('change',e=>loadFile(e.target.files[0]));

function loadFile(f){
  if(!f) return;
  uploadedFile=f; token=null; pageB64s=[];
  document.getElementById('procBtn').disabled=false;
  document.getElementById('dlPanel').classList.remove('show');
  document.getElementById('progWrap').classList.remove('show');
  document.getElementById('progBar').style.width='0%';
  document.getElementById('statusBar').textContent=`📁 ${f.name}  (${(f.size/1024).toFixed(1)} KB)`;
  if(!f.name.toLowerCase().endsWith('.pdf')){
    const r=new FileReader(); r.onload=ev=>showBig(ev.target.result); r.readAsDataURL(f);
  } else {
    document.getElementById('statusBar').textContent+=` — PDF · click ⚡ Process All Pages`;
  }
}

function selFx(el){
  document.querySelectorAll('.ebtn').forEach(b=>b.classList.remove('on'));
  el.classList.add('on'); fx=el.dataset.e;
}

function showBig(src){
  document.getElementById('ph').style.display='none';
  document.getElementById('bigPreview').style.display='flex';
  document.getElementById('bigImg').src=src;
}

function showThumb(idx){
  showBig('data:image/png;base64,'+pageB64s[idx]);
  document.querySelectorAll('.page-card img').forEach((im,i)=>im.classList.toggle('sel',i===idx));
}

async function processAll(){
  if(!uploadedFile) return;
  document.getElementById('procBtn').disabled=true;
  document.getElementById('progWrap').classList.add('show');
  document.getElementById('progLabel').textContent='Uploading…';
  document.getElementById('dlPanel').classList.remove('show');
  document.getElementById('overlay').classList.add('show');
  document.getElementById('omsg').textContent='Processing all pages…';

  const fd=new FormData();
  fd.append('file',uploadedFile);
  fd.append('auto_crop',document.getElementById('autoCrop').checked);
  fd.append('border_px',document.getElementById('bpx').value);
  fd.append('effect',fx);
  fd.append('do_ocr',document.getElementById('doOCR').checked);

  try{
    const resp=await fetch('/process_all',{method:'POST',body:fd});
    const data=await resp.json();
    if(data.error){alert('Error: '+data.error);return;}

    pageB64s=data.pages; token=data.token;

    // Render thumbnails
    document.getElementById('ph').style.display='none';
    const strip=document.getElementById('thumbStrip');
    strip.style.display='flex';
    strip.innerHTML=data.pages.map((b,i)=>
      `<div class="page-card" onclick="showThumb(${i})">
         <img src="data:image/png;base64,${b}" class="${i===0?'sel':''}">
         <span>Page ${i+1}</span>
       </div>`).join('');
    showBig('data:image/png;base64,'+data.pages[0]);

    // Info grid
    document.getElementById('infoRow').style.display='grid';
    document.getElementById('infoGrid').innerHTML=Object.entries(data.info)
      .map(([k,v])=>`<div class="iitem"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');

    // OCR
    if(data.ocr_text){
      document.getElementById('ocrCard').style.display='block';
      document.getElementById('ocrPre').textContent=data.ocr_text;
    } else { document.getElementById('ocrCard').style.display='none'; }

    // Progress complete
    document.getElementById('progBar').style.width='100%';
    document.getElementById('progLabel').textContent=`${data.pages.length} / ${data.pages.length} pages done`;

    // Download panel
    document.getElementById('dlPanel').classList.add('show');
    document.getElementById('dlInfo').textContent=`${data.pages.length} page(s) ready to download`;
    document.getElementById('statusBar').textContent=`✅ Done! ${data.pages.length} page(s) processed. Choose a download format below.`;

  }catch(err){alert('Failed: '+err.message);}
  finally{
    document.getElementById('procBtn').disabled=false;
    document.getElementById('overlay').classList.remove('show');
  }
}

function dlFile(fmt){
  if(!token){alert('Please process a file first!');return;}
  window.location.href=`/download/${fmt}/${token}`;
}

function copyOCR(){
  navigator.clipboard.writeText(document.getElementById('ocrPre').textContent).then(()=>alert('Copied!'));
}
</script>
</body>
</html>
"""

# ── routes ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return HTML_PAGE


@app.route('/process_all', methods=['POST'])
def process_all():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'})

        auto_crop = request.form.get('auto_crop', 'true').lower() == 'true'
        border_px = int(request.form.get('border_px', 0))
        effect    = request.form.get('effect', 'none')
        do_ocr    = request.form.get('do_ocr', 'false').lower() == 'true'

        file_bytes = file.read()
        filename   = file.filename.lower()
        processed_imgs = []
        ocr_parts      = []

        if filename.endswith('.pdf'):
            doc = fitz.open(stream=file_bytes, filetype='pdf')
            total = len(doc)
            for i in range(total):
                pil = pdf_page_to_pil(doc, i, dpi=180)
                pil, _ = process_image(pil, auto_crop, border_px, effect)
                processed_imgs.append(pil)
                if do_ocr:
                    ocr_parts.append(f"--- Page {i+1} of {total} ---\n{pytesseract.image_to_string(pil).strip()}")
            doc.close()
        else:
            pil = Image.open(io.BytesIO(file_bytes)).convert('RGB')
            pil, _ = process_image(pil, auto_crop, border_px, effect)
            processed_imgs.append(pil)
            if do_ocr:
                ocr_parts.append(pytesseract.image_to_string(pil).strip())

        import hashlib, time
        token = hashlib.md5(f"{file.filename}{time.time()}".encode()).hexdigest()
        _results_store[token] = processed_imgs

        # Thumbnails for preview (small)
        thumbs = []
        for img in processed_imgs:
            t = img.copy(); t.thumbnail((280, 360))
            thumbs.append(img_to_base64(t))

        out_w = processed_imgs[0].width
        out_h = processed_imgs[0].height

        result = {
            'token': token,
            'pages': thumbs,
            'info': {
                'Total Pages'  : str(len(processed_imgs)),
                'Output Size'  : f'{out_w} × {out_h} px',
                'Effect'       : effect.capitalize(),
                'Auto-cropped' : 'Yes' if auto_crop else 'No',
                'Border Added' : f'{border_px}px' if border_px else 'None',
                'File'         : file.filename,
            },
        }
        if ocr_parts:
            result['ocr_text'] = '\n\n'.join(ocr_parts)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})


@app.route('/download/<fmt>/<token>')
def download(fmt, token):
    imgs = _results_store.get(token)
    if not imgs:
        return "Session expired — please re-process the file.", 404

    # ── PDF ────────────────────────────────────────────────────────────────
    if fmt == 'pdf':
        buf = io.BytesIO()
        imgs[0].save(
            buf, format='PDF', save_all=True,
            append_images=imgs[1:], resolution=150
        )
        buf.seek(0)
        return send_file(buf, mimetype='application/pdf',
                         as_attachment=True, download_name='processed.pdf')

    # ── ZIP of PNGs ────────────────────────────────────────────────────────
    elif fmt == 'zip':
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for i, img in enumerate(imgs):
                pbuf = io.BytesIO()
                img.save(pbuf, format='PNG')
                zf.writestr(f'page_{i+1:03d}.png', pbuf.getvalue())
        zbuf.seek(0)
        return send_file(zbuf, mimetype='application/zip',
                         as_attachment=True, download_name='processed_pages.zip')

    # ── Long vertical PNG strip ────────────────────────────────────────────
    elif fmt == 'strip':
        GAP   = 12
        max_w = max(img.width  for img in imgs)
        tot_h = sum(img.height for img in imgs) + GAP * (len(imgs) - 1)
        strip = Image.new('RGB', (max_w, tot_h), (230, 230, 230))
        y = 0
        for img in imgs:
            strip.paste(img, ((max_w - img.width) // 2, y))
            y += img.height + GAP
        sbuf = io.BytesIO()
        strip.save(sbuf, format='PNG')
        sbuf.seek(0)
        return send_file(sbuf, mimetype='image/png',
                         as_attachment=True, download_name='processed_strip.png')

    return "Unknown format", 400


if __name__ == '__main__':
    print("=" * 55)
    print("  Image & PDF Processor  (PyMuPDF — no Poppler needed)")
    print("  Open: http://localhost:5000")
    print("=" * 55)
    app.run(debug=False, host='0.0.0.0', port=5000)
