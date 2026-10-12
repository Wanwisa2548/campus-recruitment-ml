"""Shared Task 2 mappings and local demo. Importing never trains or loads a model."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageOps, UnidentifiedImageError
from IPython.display import display

IMG_SIZE = (224, 224)
DEMO_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
final_model = None
tf = None
thai_font = None

class_names = ["tom_yum", "som_tam", "noodle", "larb", "pad_kra_pao"]
thai_labels = {
    "tom_yum": "ต้มยำ", "som_tam": "ส้มตำ", "noodle": "ก๋วยเตี๋ยว",
    "larb": "ลาบ", "pad_kra_pao": "ผัดกะเพรา",
}
english_labels = {
    "tom_yum": "Tom Yum", "som_tam": "Som Tam", "noodle": "Noodle",
    "larb": "Larb", "pad_kra_pao": "Pad Kra Pao",
}

SHOW_CALORIES = True
calorie_info = {
    "tom_yum": "ประมาณ 100–300 kcal / 1 ถ้วย (ไม่รวมข้าว; น้ำใส/น้ำข้นต่างกัน)",
    "som_tam": "ประมาณ 80–200 kcal / 1 จาน (ไม่รวมข้าวเหนียวและเครื่องเคียง)",
    "noodle": "ประมาณ 250–500 kcal / 1 ชาม (ขึ้นกับเส้น เนื้อ และน้ำซุป)",
    "larb": "ประมาณ 150–350 kcal / 1 จาน (ไม่รวมข้าวเหนียว)",
    "pad_kra_pao": "ประมาณ 450–750 kcal / 1 จานพร้อมข้าว (ไม่รวมไข่ดาว)",
}

CALORIE_DISCLAIMER = (
    "Estimated calories are approximate reference values based on a typical serving. "
    "The image classification model does not measure food weight, ingredients, portion size, or actual calories."
)

def predict_food(image_path):
    """ทำนายภาพใหม่ด้วยโมเดลเดิม; EXIF -> RGB -> bilinear resize -> raw pixels."""
    if globals().get("final_model") is None:
        print("ยังไม่มีโมเดลใน kernel — รันเซลล์ Presentation Mode เพื่อโหลดโมเดลเดิม")
        return None
    path = Path(image_path)
    if not path.is_file():
        print(f"ไม่พบรูป: {path}")
        return None
    # ไม่แก้ IMAGE_EXTENSIONS ของ Dataset; WEBP รองรับเฉพาะการทำนายภาพใหม่ด้วย Pillow
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
        print("กรุณาใช้ JPG / JPEG / PNG / WEBP / BMP (หรือ GIF เฟรมแรก)")
        return None
    try:
        with Image.open(path) as img:
            original = ImageOps.exif_transpose(img).convert("RGB")
            original.load()
            raw_pixels = np.asarray(original, dtype="float32")
        resized = tf.image.resize(raw_pixels, IMG_SIZE, method="bilinear")
        batch = tf.expand_dims(resized, axis=0)
        # preprocessing อยู่ในโมเดลแล้ว: ไม่หาร 255 และไม่เรียก preprocess_input ซ้ำ
        # เรียก inference โดยตรงสำหรับภาพเดียว เพื่อลด predict-wrapper retracing/logs
        probabilities = np.asarray(final_model(batch, training=False))[0]
        predicted_index = int(np.argmax(probabilities))
        name = class_names[predicted_index]
        confidence = float(probabilities[predicted_index] * 100)
        calories = calorie_info.get(name) if SHOW_CALORIES else None
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        print("อ่านรูปหรือทำนายไม่สำเร็จ กรุณาเลือกภาพที่อ่านได้:", exc)
        return None
    except Exception as exc:
        print("ทำนายไม่สำเร็จ กรุณาตรวจโมเดลใน kernel แล้วลองใหม่:", exc)
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(original)
    ax.axis("off")
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    print(f"Food: {english_labels[name]} ({thai_labels[name]})")
    print(f"Class: {name}")
    print(f"Confidence: {confidence:.2f}%")
    if SHOW_CALORIES:
        print(f"Estimated Calories: {calories or 'ยังไม่กำหนดค่าอ้างอิง'}")
    return {"class": name, "thai_name": thai_labels[name], "confidence_percent": confidence,
            "estimated_calories": calories}

def _predict_demo_image(selected_path):
    """Cancel และไฟล์ที่ไม่รองรับไม่เรียกโมเดล; ผลแสดงโดย predict_food เพียงครั้งเดียว."""
    if not selected_path:
        print("ยกเลิกการเลือกรูป — เลือกใหม่ได้เมื่อพร้อม")
        return None
    if Path(selected_path).suffix.lower() not in DEMO_IMAGE_EXTENSIONS:
        print("กรุณาเลือก JPG, JPEG, PNG, WEBP หรือ BMP")
        return None
    return predict_food(selected_path)

def choose_food_image():
    """เปิด Windows file picker โดยไม่พึ่ง training cells."""
    if globals().get("final_model") is None or not callable(globals().get("predict_food")):
        print("กรุณารัน Presentation Mode เพื่อโหลดโมเดลเดิมก่อนเลือกรูป")
        return None
    root = None
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        selected_path = filedialog.askopenfilename(
            parent=root,
            title="เลือกรูปอาหารใหม่ (ไม่ใช่ Train / Validation / Test)",
            filetypes=[
                ("Food images", "*.jpg *.jpeg *.png *.webp *.bmp *.JPG *.JPEG *.PNG *.WEBP *.BMP"),
                ("JPG / JPEG", "*.jpg *.jpeg *.JPG *.JPEG"),
                ("PNG", "*.png *.PNG"),
                ("WEBP", "*.webp *.WEBP"),
                ("BMP", "*.bmp *.BMP"),
            ],
        )
    except Exception as exc:
        print("เปิดหน้าต่างเลือกไฟล์ไม่ได้:", exc)
        print("ใช้ local Windows kernel ที่มี tkinter และหน้าจอ แล้วลองใหม่")
        return None
    finally:
        if root is not None:
            try:
                root.destroy()
            except tk.TclError:
                pass
    return _predict_demo_image(selected_path)

def show_food_picker():
    """แสดงปุ่ม; ถ้าไม่มี widget ให้ใช้ choose_food_image() โดยตรง."""
    if globals().get("final_model") is None or not callable(globals().get("predict_food")):
        print("ยังไม่พร้อมสาธิต — รัน Presentation Mode ท้าย Notebook ก่อน")
        return
    try:
        import ipywidgets as widgets
    except ImportError:
        print("ไม่มี ipywidgets — รันเซลล์ fallback: choose_food_image()")
        return

    button = widgets.Button(
        description="เลือกรูปอาหาร", icon="folder-open", button_style="info",
        tooltip="เลือก JPG, JPEG, PNG, WEBP หรือ BMP จากเครื่อง",
        layout=widgets.Layout(width="200px"),
    )
    output = widgets.Output()

    def on_click(clicked_button):
        clicked_button.disabled = True
        try:
            with output:
                # ล้างเฉพาะพื้นที่ผล Demo ของปุ่มนี้ ไม่ล้างผลฝึก/ประเมินเดิม
                output.clear_output(wait=True)
                choose_food_image()
        finally:
            clicked_button.disabled = False

    button.on_click(on_click)
    display(button, output)
    print("กดเลือกรูปอาหาร หากปุ่มไม่แสดง ให้รันเซลล์ fallback ด้านล่าง")


def start_demo(model=None):
    """Use the in-memory model, or load the saved model after a kernel restart."""
    global final_model, tf, thai_font
    final_model = None
    try:
        import tensorflow as tf

        installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
        thai_font = next((name for name in ["Tahoma", "Noto Sans Thai", "Leelawadee UI", "Th Sarabun New"]
                          if name in installed_fonts), None)
        if thai_font:
            plt.rcParams["font.family"] = thai_font
        plt.rcParams["axes.unicode_minus"] = False

        if model is None:
            model_path = Path("thai_food_model.keras")
            if not model_path.is_file():
                print("ไม่พบ thai_food_model.keras — วางไฟล์โมเดลเดิมในโฟลเดอร์โปรเจกต์")
                print("Current working directory:", Path.cwd())
                return None
            model = tf.keras.models.load_model(model_path, compile=False)
        if tuple(model.input_shape[1:]) != (*IMG_SIZE, 3):
            raise ValueError(f"Input shape ไม่ตรงกับ 224×224 RGB: {model.input_shape}")
        if tuple(model.output_shape[1:]) != (len(class_names),):
            raise ValueError(f"Output shape ไม่ตรงกับ 5 คลาส: {model.output_shape}")
        final_model = model
    except Exception as exc:
        print("เตรียม Demo ไม่สำเร็จ:", exc)
        print("ตรวจไฟล์โมเดลและเลือก environment ที่เคยใช้งานได้ แล้วลองใหม่")
        return None

    print("โมเดลเดิมพร้อมสาธิต — ไม่มีการฝึกหรือบันทึก weights ใหม่")
    try:
        show_food_picker()
    except Exception as exc:
        print("ปุ่ม widget ไม่พร้อมใช้งาน:", exc)
        print("โมเดลยังพร้อมทำนาย — ใช้เซลล์ fallback ใต้ Presentation Mode")
    return final_model
