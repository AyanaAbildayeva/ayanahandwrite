import easyocr
import numpy as np
import cv2
import base64
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


reader = easyocr.Reader(['en', 'ru'], gpu=False)

@app.route('/')
def index():
    return render_template('handsite.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['image']
        encoded = data.split(',')[1]
        img_bytes = base64.b64decode(encoded)
        
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # --- СУРЕТТІ ЖАҚСАРТУ 
       
        img = cv2.convertScaleAbs(img, alpha=1.8, beta=-20)
        
        # 2. Шуды кетіру 
        img = cv2.medianBlur(img, 3)

        # --- ТАНУ (EasyOCR) 
       
        results = reader.readtext(img, detail=0, paragraph=True)
        
        final_text = " ".join(results)
        print(f"Танылды: {final_text}")

        return jsonify({
            'success': True,
            'text': final_text if final_text.strip() else "Ештеңе танылмады. Анығырақ жазыңыз."
        })

    except Exception as e:
        print(f"Қате: {e}")
        return jsonify({'success': False, 'text': 'Серверде қате орын алды'})

if __name__ == '__main__':
   
    app.run(debug=True)