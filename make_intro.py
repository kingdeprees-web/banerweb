#!/usr/bin/env python3
"""
ساخت اینترو کمیته از روی عکس بنر (انجمن سازندگان ذی‌صلاح ارومیه)

  python make_intro.py banners/legal.jpg --out videos
  python make_intro.py --all banners --out videos

اگر کنار بنر فایل هم‌نام .txt باشد (مثلاً banners/legal.txt) اسم‌ها از آن خوانده می‌شود:
  عنوان: کارشناسی و حقوقی
  فرشاد موسوی مقدم | لیسانس عمران | کارشناس دادگستری
  ناصر عادلی | کارشناس ارشد عمران | کارشناس دادگستری
در غیر این صورت (اگر tesseract فارسی نصب باشد) اسم‌ها خودکار خوانده می‌شود.
"""
import argparse, glob, json, os, shutil, subprocess, sys, tempfile
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- تشخیص لوگو و عکس اعضا
def detect(img):
    """img: BGR. خروجی: (logo, members) هرکدام dict با x,y,r بر حسب پیکسل بنر"""
    H, W = img.shape[:2]
    f = max(1, round(W / 556))
    sm = img[f // 2::f, f // 2::f][:, :, ::-1].astype(np.int32)  # RGB
    r, g, b = sm[..., 0], sm[..., 1], sm[..., 2]
    gold = ((r > 140) & (r >= g) & (g > b) & (r - b > 70) & (g > 85)).astype(np.uint8)
    h, w = gold.shape
    n, _, st, _ = cv2.connectedComponentsWithStats(gold, connectivity=8)
    logo, mem = None, []
    for i in range(1, n):
        x0, y0, bw, bh, cnt = st[i]
        sz = (bw + bh) / 2
        if abs(bw - bh) / max(bw, bh) > 0.12 or cnt < sz * 3:
            continue
        cx, cy, rel = (x0 + bw / 2) * f, (y0 + bh / 2) * f, sz / w
        if 0.2 < rel < 0.34 and cy < H * 0.3:
            if logo is None or sz > logo['sz']:
                logo = dict(x=cx, y=cy, r=sz / 2 * f, sz=sz)
        elif 0.10 < rel < 0.19 and cx > W * 0.7:
            mem.append(dict(x=cx, y=cy, r=sz / 2 * f))
    mem.sort(key=lambda m: m['y'])
    if logo:
        mem = [m for m in mem if m['y'] > logo['y'] + logo['r'] * 1.1]
    out = []
    for m in mem:
        if not out or m['y'] - out[-1]['y'] > m['r']:
            out.append(m)
    return logo, out


# ---------------------------------------------------------------- خواندن متن (اختیاری)
def fix_fa(s):
    return s.replace('ي', 'ی').replace('ك', 'ک').replace('|', '').strip()


def ocr_names(img, mem):
    try:
        import pytesseract
        from PIL import Image
        pytesseract.get_languages()
        if 'fas' not in pytesseract.get_languages():
            return None
    except Exception:
        return None
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def rd(x, y, w, h, th, psm):
        c = gray[int(max(0, y)):int(y + h), int(x):int(x + w)]
        c = cv2.resize(c, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        c = np.where(c < th, 0, 255).astype(np.uint8)
        t = pytesseract.image_to_string(c, lang='fas', config=f'--psm {psm}')
        return [fix_fa(z) for z in t.split('\n') if z.strip()]

    title = ' '.join(rd(.07 * W, mem[0]['y'] - .186 * W, .55 * W, .06 * W, 115, 7))
    rows = []
    for m in mem:
        n = rd(.551 * W, m['y'] - .02 * W, .227 * W, .092 * W, 150, 6)
        c = rd(.308 * W, m['y'] - .02 * W, .225 * W, .092 * W, 150, 6)
        role = (n[1] if len(n) > 1 else (c[1] if len(c) > 1 else '')).replace('(', '').replace(')', '').strip()
        rows.append(dict(n=n[0] if n else '', c=c[0] if c else '', r=role))
    return title, rows


def read_txt(path):
    title, rows = '', []
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('عنوان:') or line.lower().startswith('title:'):
            title = line.split(':', 1)[1].strip()
            continue
        p = [x.strip() for x in line.split('|')]
        rows.append(dict(n=p[0], c=p[1] if len(p) > 1 else '', r=p[2] if len(p) > 2 else ''))
    return title, rows


# ---------------------------------------------------------------- ساخت ویدیو
def run(cmd, **kw):
    r = subprocess.run(cmd, **kw)
    if r.returncode:
        sys.exit(f'خطا در اجرای: {" ".join(map(str, cmd))}')


def build(banner, out_dir, title_arg, workers):
    name = os.path.splitext(os.path.basename(banner))[0]
    img = cv2.imread(banner)
    if img is None:
        sys.exit(f'نمی‌توانم عکس را باز کنم: {banner}')
    logo, mem = detect(img)
    if not logo or not mem:
        sys.exit(f'[{name}] لوگو یا عکس اعضا پیدا نشد؛ مطمئن شو بنر کمیته با همان قالب انجمن است.')
    print(f'[{name}] {len(mem)} عضو پیدا شد')

    title, rows = '', None
    txt = os.path.splitext(banner)[0] + '.txt'
    if os.path.exists(txt):
        title, rows = read_txt(txt)
        print(f'[{name}] اسم‌ها از {os.path.basename(txt)} خوانده شد')
    if rows is None:
        res = ocr_names(img, mem)
        if res:
            title, rows = res
            print(f'[{name}] اسم‌ها با OCR خوانده شد (حتماً بررسی کن)')
    if rows is None:
        tpl = txt
        with open(tpl, 'w', encoding='utf-8') as f:
            f.write('عنوان: \n' + ''.join('نام | مدرک | توضیح اختیاری\n' for _ in mem))
        sys.exit(f'[{name}] OCR فارسی در دسترس نیست. فایل {tpl} را با اسم‌ها پر کن و دوباره اجرا کن.')
    if title_arg:
        title = title_arg
    rows = (rows + [dict(n='', c='', r='')] * len(mem))[:len(mem)]
    if len(rows) != len(mem):
        print(f'[{name}] هشدار: تعداد اسم‌ها با تعداد عکس‌ها فرق دارد')

    work = tempfile.mkdtemp(prefix='intro_')
    cfg = dict(banner=os.path.abspath(banner), title=title or 'کمیته تخصصی',
               members=rows, circles=mem, logo=logo)
    cfgp = os.path.join(work, 'cfg.json')
    json.dump(cfg, open(cfgp, 'w', encoding='utf-8'), ensure_ascii=False)
    env = dict(os.environ, INTRO_CFG=cfgp)
    eng = os.path.join(HERE, 'engine.py')

    info = subprocess.run([sys.executable, eng, 'info'], env=env, capture_output=True, text=True, check=True).stdout
    I = json.loads(info.strip().splitlines()[-1])
    total = int(I['TEND'] * I['FPS'])
    print(f'[{name}] مدت ویدیو {I["TEND"]:.1f} ثانیه ({total} فریم)')

    # رندر موازی قطعه‌ها
    chunk = max(30, total // (workers * 2))
    parts, procs = [], []
    a = 0
    while a < total:
        b = min(total, a + chunk)
        parts.append((a, b, os.path.join(work, f'p_{len(parts):03d}.mp4')))
        a = b
    pending, running = list(parts), []
    while pending or running:
        while pending and len(running) < workers:
            p = pending.pop(0)
            running.append((subprocess.Popen([sys.executable, eng, str(p[0]), str(p[1]), p[2]], env=env), p))
        for pr, p in list(running):
            if pr.poll() is not None:
                if pr.returncode:
                    sys.exit('رندر قطعه ناموفق بود')
                running.remove((pr, p))
                done = sum(1 for q in parts if os.path.exists(q[2]) and q not in [x[1] for x in running] and q not in pending)
                print(f'  رندر {done}/{len(parts)} قطعه')
        import time; time.sleep(0.5)
    lst = os.path.join(work, 'list.txt')
    open(lst, 'w').write(''.join(f"file '{p[2]}'\n" for p in parts))
    silent = os.path.join(work, 'silent.mp4')
    run(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', silent])

    # صدا
    run([sys.executable, os.path.join(HERE, 'audio_gen.py'), json.dumps(I), work])
    os.makedirs(out_dir, exist_ok=True)
    for kind, wav, lufs in (('', 'music.wav', -14), ('_NoMusic', 'sfx.wav', -16)):
        outp = os.path.join(out_dir, f'Intro_{name}{kind}.mp4')
        run(['ffmpeg', '-loglevel', 'error', '-y', '-i', silent, '-i', os.path.join(work, wav), '-map', '0:v', '-map', '1:a',
             '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k', '-af', f'loudnorm=I={lufs}:TP=-1.5:LRA=11', '-shortest', outp])
        print(f'[{name}] ساخته شد: {outp}')
    shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('banner', nargs='?', help='عکس بنر کمیته')
    ap.add_argument('--all', metavar='DIR', help='همه بنرهای یک پوشه')
    ap.add_argument('--out', default='videos')
    ap.add_argument('--title', help='عنوان کمیته (اختیاری)')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2)))
    a = ap.parse_args()
    if a.all:
        files = sorted(p for e in ('jpg', 'jpeg', 'png', 'webp') for p in glob.glob(os.path.join(a.all, f'*.{e}')))
        if not files:
            sys.exit('هیچ بنری در پوشه نیست')
        for f in files:
            build(f, a.out, a.title, a.workers)
    elif a.banner:
        build(a.banner, a.out, a.title, a.workers)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
