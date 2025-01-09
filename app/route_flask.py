from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from handlers import upload_image_handler, get_image_handler

router = Blueprint('router', __name__)

@router.route("/upload/", methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    setting = request.form.get('setting')
    user_id = request.form.get('user_id')
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith(('png', 'jpg', 'jpeg')):
        try:
            file_path = upload_image_handler(file, setting, user_id)
            return jsonify({"message": "تصویر با موفقیت پردازش شد", "file_path": file_path}), 200
        except Exception as e:
            return jsonify({"error": f"آپلود تصویر با خطا مواجه شد: {str(e)}"}), 500
    else:
        return jsonify({"error": "File is not an image"}), 400

@router.route("/image/<filename>", methods=['GET'])
def get_image(filename):
    try:
        image_path = get_image_handler(filename)
        return jsonify({"image_path": image_path}), 200
    except Exception as e:
        return jsonify({"error": f"دریافت تصویر با خطا مواجه شد: {str(e)}"}), 500