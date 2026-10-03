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
