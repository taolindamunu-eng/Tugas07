import cv2, os, re
import numpy as np
import pytesseract
from pytesseract import Output

FOLDER = "gambar"
GROUND_TRUTH = "571012022000056"

TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
]

for path in TESSERACT_PATHS:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        break
else:
    print("ERROR: Tesseract tidak ditemukan!")
    print("Install Tesseract terlebih dahulu.")
    exit()


def contrast_stretch(img):
    mn, mx = np.min(img), np.max(img)
    if mx == mn:
        return img
    return ((img - mn) * 255 / (mx - mn)).astype(np.uint8)


def clahe(img):
    return cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(img)


def gamma_correction(img, gamma=1.5):
    table = np.array([
        ((i / 255.0) ** (1 / gamma)) * 255
        for i in range(256)
    ]).astype(np.uint8)
    return cv2.LUT(img, table)


def ekstrak_nomor(text):
    text = text.replace(" ", "").replace("\n", "")
    kandidat = re.findall(r"\d{10,}", text)
    if not kandidat:
        return ""
    return min(
        kandidat,
        key=lambda x: abs(len(x) - len(GROUND_TRUTH))
    )


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(
                curr[-1] + 1,
                prev[j] + 1,
                prev[j - 1] + cost
            ))
        prev = curr
    return prev[-1]


def hitung_cer(prediksi):
    if not prediksi:
        return 1.0
    return levenshtein(
        prediksi, GROUND_TRUTH
    ) / len(GROUND_TRUTH)


def buat_text_mask(img):
    data = pytesseract.image_to_data(
        img,
        config="--psm 6",
        output_type=Output.DICT
    )
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for i, text in enumerate(data["text"]):
        if not text.strip():
            continue

        try:
            conf = float(data["conf"][i])
        except:
            conf = 0

        if conf < 20:
            continue

        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]
        p = 5

        mask[
            max(0, y-p):min(img.shape[0], y+h+p),
            max(0, x-p):min(img.shape[1], x+w+p)
        ] = 255

    return mask


def deteksi_metode(gray, text_mask, adaptive=False):
    if adaptive:
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31, 15
        )
    else:
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    binary[text_mask == 255] = 0

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )

    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        kernel, iterations=1
    )
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        kernel, iterations=2
    )

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, 8
    )

    h, w = gray.shape
    total = h * w
    candidates = []

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        if area < total * 0.0003 or bw < w * 0.08:
            continue

        aspect = bw / max(bh, 1)
        density = area / max(bw * bh, 1)
        score = 0

        if bw >= w * 0.15:
            score += 2
        if aspect >= 2:
            score += 2
        if area >= total * 0.001:
            score += 1
        if density < 0.50:
            score += 1

        if score >= 4:
            candidates.append((area, score))

    if not candidates:
        return "ABSENT", 0.0, 0

    area, score = max(
        candidates,
        key=lambda x: (x[1], x[0])
    )

    return "PRESENT", area / total * 100, score


def deteksi_tanda_tangan(gray):
    mask = buat_text_mask(gray)

    otsu = deteksi_metode(
        gray, mask
    )
    adaptive = deteksi_metode(
        gray, mask, True
    )

    if adaptive[2] > otsu[2]:
        return adaptive[0], adaptive[1], "Adaptive"

    if otsu[2] > adaptive[2]:
        return otsu[0], otsu[1], "Otsu"

    if adaptive[1] > otsu[1]:
        return adaptive[0], adaptive[1], "Adaptive"

    return otsu[0], otsu[1], "Otsu"


hasil = []

files = sorted([
    f for f in os.listdir(FOLDER)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
])

for filename in files:
    path = os.path.join(FOLDER, filename)
    img = cv2.imread(path)

    if img is None:
        continue

    gray = cv2.cvtColor(
        img, cv2.COLOR_BGR2GRAY
    )

    metode = {
        "Contrast Stretching":
            contrast_stretch(gray),
        "CLAHE":
            clahe(gray),
        "Gamma Correction":
            gamma_correction(gray)
    }

    hasil_ocr = []

    for nama, enhanced in metode.items():
        text = pytesseract.image_to_string(
            enhanced,
            config="--psm 6"
        )
        nomor = ekstrak_nomor(text)
        cer = hitung_cer(nomor)
        hasil_ocr.append(
            (nama, nomor, cer)
        )

    terbaik = min(
        hasil_ocr,
        key=lambda x: x[2]
    )

    signature, percentage, metode_sig = \
        deteksi_tanda_tangan(gray)

    hasil.append([
        filename,
        terbaik[1],
        terbaik[2] * 100,
        terbaik[0],
        signature,
        percentage,
        metode_sig
    ])


print("\n" + "=" * 110)
print(
    f"{'Nama gambar':<35}"
    f"{'Hasil OCR':<20}"
    f"{'CER':<15}"
    f"{'Enhancement terbaik':<25}"
    f"{'Tanda tangan':<18}"
)
print("=" * 110)

for row in hasil:
    print(
        f"{row[0]:<35}"
        f"{row[1]:<20}"
        f"{row[2]:<15}"
        f"{row[3]:<25}"
        f"{row[4]:<18}"
    )

print("=" * 110)