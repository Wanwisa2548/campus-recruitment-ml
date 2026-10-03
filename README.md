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
