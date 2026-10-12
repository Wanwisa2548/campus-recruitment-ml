# งานที่ 1: ทำนายการได้งานของนักศึกษา (Campus Recruitment)

เปรียบเทียบ 3 โมเดล Machine Learning ได้แก่ **Logistic Regression, Random Forest และ SVM**
ด้วยตัวชี้วัด **Accuracy, Precision, Recall และ F1-Score**

## ไฟล์ในโปรเจกต์

| ไฟล์ | คืออะไร |
|---|---|
| `task1_campus_recruitment.ipynb` | โค้ดทั้งหมดพร้อมผลลัพธ์และกราฟ |
| `Placement_Data_Full_Class.csv` | ชุดข้อมูลจาก Kaggle (215 คน) |
| `requirements.txt` | รายชื่อไลบรารีที่ต้องติดตั้ง |

## ชุดข้อมูล

Kaggle: [Campus Recruitment](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement) โดย Ben Roshan

## วิธีรันบนเครื่องตัวเอง (Windows)

1. ติดตั้ง Python 3.10 หรือ 3.11 จาก python.org (ติ๊ก Add to PATH)
2. ดาวน์โหลดโปรเจกต์นี้ (ปุ่ม **Code > Download ZIP**) แล้วแตกไฟล์
3. เปิด Command Prompt ในโฟลเดอร์โปรเจกต์ แล้วพิมพ์

```
python -m venv ml-env
ml-env\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

4. เปิดไฟล์ `task1_campus_recruitment.ipynb` แล้วกด **Run All**

> ไฟล์ CSV ต้องอยู่โฟลเดอร์เดียวกับ Notebook

## ผลลัพธ์สรุป (ชุดทดสอบ 43 คน)

| โมเดล | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | **0.8605** | **0.9286** | 0.8667 | **0.8966** |
| Random Forest | 0.8372 | 0.8485 | **0.9333** | 0.8889 |
| SVM | 0.8372 | 0.9259 | 0.8333 | 0.8772 |

- โมเดลที่ดีที่สุดโดยรวม: **Logistic Regression**
- ปัจจัยสำคัญที่สุด: คะแนนมัธยมต้น (ssc_p), เกรดปริญญาตรี (degree_p), คะแนนมัธยมปลาย (hsc_p)
- Cross-Validation 5 รอบ: ทั้ง 3 โมเดลใกล้เคียงกัน (Accuracy ราว 0.86)

## สมาชิกกลุ่ม

- [ชื่อ-นามสกุล] [รหัสนักศึกษา]
- [ชื่อ-นามสกุล] [รหัสนักศึกษา]
- [ชื่อ-นามสกุล] [รหัสนักศึกษา]



## งานที่ 2: Thai Food Image Classification Using Computer Vision

**Objective:** จำแนกภาพอาหารไทย 5 ประเภทด้วย Computer Vision และ Transfer Learning
อินพุตเป็นรูปภาพ ไม่ใช่ข้อมูลตาราง ใช้ **MobileNetV2** ที่มี ImageNet pretrained weights

**Classes:** Tom Yum (ต้มยำ), Som Tam (ส้มตำ), Noodle (ก๋วยเตี๋ยว), Larb (ลาบ),
Pad Kra Pao (ผัดกะเพรา) โดยชื่อโฟลเดอร์คือ `tom_yum`, `som_tam`, `noodle`, `larb`, `pad_kra_pao`

**Notebook:** `task2_thai_food_classification.ipynb` มีคำอธิบายภาษาไทยและขั้นตอนตั้งแต่ตรวจภาพ
สำรวจข้อมูล, Augmentation, Base Transfer Learning, Optional Fine-Tuning จนถึงประเมินและทำนายภาพใหม่

### เตรียม Dataset และรันงานที่ 2

1. ใช้ Python environment ที่ติดตั้ง `python -m pip install -r requirements.txt` แล้ว
   แนะนำ Python 3.11 เช่นเดียวกับงานที่ 1 เปิด Jupyter จากโฟลเดอร์โปรเจกต์และเลือก kernel ให้ตรงกัน
2. วาง **รูปอาหารจริง** ในโฟลเดอร์ด้านล่าง แนะนำรวม 100–200 รูปต่อคลาส
   แบ่ง **Train 70% / Validation 15% / Test 15%** เช่น 200 รูป → 140 / 30 / 30 ต่อคลาส
   ใช้ JPG/JPEG/PNG/BMP/GIF; แปลง HEIC จากโทรศัพท์เป็น JPG ก่อน

```text
thai_food/
├── train/
│   ├── tom_yum/
│   ├── som_tam/
│   ├── noodle/
│   ├── larb/
│   └── pad_kra_pao/
├── val/
│   ├── tom_yum/
│   ├── som_tam/
│   ├── noodle/
│   ├── larb/
│   └── pad_kra_pao/
└── test/
    ├── tom_yum/
    ├── som_tam/
    ├── noodle/
    ├── larb/
    └── pad_kra_pao/
```

3. เปิด Notebook แล้ว **Restart Kernel → Run All** โค้ดใช้ Relative paths ทั้งหมด
   นับภาพอัตโนมัติและแจ้งเมื่อโฟลเดอร์ว่าง/หายหรือรูปเสีย โดยข้ามการฝึกหากยังไม่พร้อม
   `.gitkeep` เก็บโฟลเดอร์ว่างไว้ใน Git และไม่ถูกนับเป็นรูป ไม่ต้องลบเมื่อเติมภาพ
4. ฝึกเริ่มต้นไม่เกิน 10 Epochs, Batch size 16 ใช้ CPU ได้
   Keras อาจดาวน์โหลด ImageNet weights ครั้งแรกเมื่อ Dataset พร้อม แต่ไม่ดาวน์โหลดรูปอาหาร
   บันทึก Base model ที่ Validation loss ดีที่สุดเป็น `thai_food_model.keras`
   Optional Fine-Tuning ปิดไว้โดยค่าเริ่มต้น และเก็บ checkpoint แยกหากเปิดใช้
5. ประเมินด้วย Test หลังเลือกโมเดลจาก Validation เท่านั้น:
   **Accuracy, Precision, Recall, F1-Score, Classification report และ Confusion Matrix**
   จากนั้นวางภาพใหม่จากโทรศัพท์เป็น `my_food_test.jpg` และเรียก `predict_food(image_path)`
   เพื่อแสดงประเภทอาหารภาษาไทย/อังกฤษและ Confidence score

**Data leakage:** ห้ามวางภาพเดียวกันหรือ near-duplicate ข้าม Train/Validation/Test
ภาพของจานเดียวกันจากชุดถ่ายเดียวกันควรอยู่ split เดียว Notebook ตรวจไฟล์ซ้ำข้าม split ด้วย SHA-256
แต่ภาพ crop/resize/บีบอัดใหม่ยังต้องตรวจเอง ไม่ใช้ Test ในการฝึกหรือปรับ Hyperparameters

**Preprocessing:** Resize เป็น 224×224 RGB แล้วให้โมเดลเรียก MobileNetV2 `preprocess_input` เพียงครั้งเดียว
การฝึก ประเมิน และรูปใหม่จึงใช้การแปลงพิกเซลเดียวกัน ห้ามหาร 255 ซ้ำ
อ้างอิง [MobileNetV2 preprocessing](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2/preprocess_input)

**Optional application:** แสดงช่วงแคลอรีอ้างอิงต่อหน่วยบริโภคด้วย Dictionary ที่แก้ไขได้
ตั้ง `SHOW_CALORIES=False` เพื่อปิด ค่าตั้งต้นเป็นตัวอย่างประมาณการสำหรับการศึกษา
ไม่ใช่ผลวิเคราะห์พลังงานจากรูป ควรปรับตามสูตรและหน่วยบริโภคจริง

> Estimated calories are approximate reference values based on a typical serving. The image classification model does not measure food weight, ingredients, portion size, or actual calories.

**สถานะ:** โฟลเดอร์ Dataset เริ่มต้นว่าง ต้องเติมภาพก่อนฝึก ยังไม่มีผลโมเดลจริงหรือ Accuracy สมมติ
เมื่อฝึกด้วยรูปจริงแล้วให้บันทึก Notebook พร้อม Outputs สำหรับการส่งงาน

## Task 2 - Dataset Preparation

Use this automatic workflow instead of manually distributing photos into the split folders described above.

1. Install the existing dependencies: `python -m pip install -r requirements.txt`.
2. Put collected photos directly into the matching folders below. Aim for **100-150 unique valid images per class**. Supported inputs: JPG, JPEG, PNG, WebP, and BMP. Convert HEIC first. Files in nested raw subfolders are not scanned.

   ```text
   thai_food_raw/
     larb/
     noodle/
     pad_kra_pao/
     som_tam/
     tom_yum/
     sources.csv
   ```

3. From the project folder, run:

   ```sh
   python dataset_prepare.py
   ```

   The script validates images with Pillow, skips corrupted files, applies EXIF orientation, converts to RGB JPEG, and removes duplicates **before** splitting. It checks SHA-256 of files and decoded pixels, plus conservative perceptual hashes across all five classes. Cross-class duplicates keep the first occurrence in the class order above; review any reported label conflicts.

4. Find the output in `thai_food/train/<class>/`, `thai_food/val/<class>/`, and `thai_food/test/<class>/`. Each class uses **70% training / 15% validation / 15% test**, with seed **42**. Integer rounding uses largest remainders. Classes with at least three unique photos get at least one in every split; one or two photos cannot fill every split. Fewer than 50 unique valid images triggers a warning. The report includes raw, valid, corrupted, duplicate, and split counts; valid means usable **after** deduplication.
5. Open `task2_thai_food_classification.ipynb` from the project folder.
6. Select **Restart Kernel**.
7. Select **Run All**.
8. Train **MobileNetV2** using the notebook's training cells. The notebook applies `preprocess_input` once inside the model; do not divide pixels by 255 separately.
9. Evaluate the **test dataset** after selecting the model using validation data.
10. Try `predict_food("my_food_test.jpg")` with a new real-world JPG photo.

**Safe reruns:** The script announces the output it will regenerate and records its generated JPGs and hashes in `thai_food/.dataset_prepare_manifest.json`. It only removes images recorded there. It preserves raw images, `.gitkeep`, notes, source code, notebooks, and unrelated files. If a generated image was edited or an unmanaged image already exists in a split folder, preparation stops and preserves it. Move unmanaged photos outside `thai_food/` before running; do not use existing training/validation/test photos as raw source data. Keep the manifest for safe future runs. If raw folders become empty or all images are unreadable, an existing generated dataset is preserved.

**Prevent leakage:** Never use test images for training, augmentation, or hyperparameter selection. Perceptual hashing is a heuristic: review candidates for missed crops, screenshots, and different views of the same dish. It may also flag similar-looking unrelated photos. Keep related photo sessions in one split or retain only one representative before preparation. No training metrics or predictions are supplied by these preparation tools.

### Optional download helper and source tracking

The helper uses **user-configured direct image URLs**, without scraping Google Images. No API key is required. Copy `food_image_urls.example.csv` to `food_image_urls.csv`, then add one row per candidate with these columns:

```text
class,source_url,source_provider,license,notes
```

Use only the five class names above. Obtain actual image-file URLs from a source that permits downloading, or export URLs and metadata from an API you are authorized to use. `source_provider`, `license`, and `notes` can be empty; an empty license means unknown. If you use an external API to obtain URLs, follow its setup instructions and store any credentials in environment variables; do not put keys in CSVs or Git. This helper has no built-in API integration.

```sh
python download_food_images.py --search-terms
python download_food_images.py --urls food_image_urls.csv --target 125
python dataset_prepare.py
```

The searches include predefined English and Thai terms for every class. The downloader aims for 125 candidates per class (adjust to 100-150), uses a 20-second timeout and 20 MB limit per file, pauses between requests, reports failures and progress, skips repeated URLs, and avoids overwriting files. It needs enough supplied URLs to reach the target. Review downloads for correct food labels before preparing. Running it without `--urls` prints setup guidance and exits without downloading.

Downloaded metadata is appended to `thai_food_raw/sources.csv` using `filename,class,source_url,source_provider,license,notes`. Filenames are relative to `thai_food_raw/`. For manually collected images, add rows to the same file yourself. Raw filenames remain unchanged; normalized output names look like `larb_0001.jpg`.

**Downloaded images are not automatically copyright-free. Check image licenses and source terms before use or redistribution.**

To verify the preparation utilities without training a model or contacting image sources:

```sh
python -m unittest discover -s tests -v
```

Tests use temporary synthetic images only; they do not populate the real food dataset or produce model results.
