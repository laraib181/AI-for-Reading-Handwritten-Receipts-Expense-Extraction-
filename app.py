from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from models.ocr_utils import process_receipt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        file = request.files['receipt']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            result = process_receipt(filepath)

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)