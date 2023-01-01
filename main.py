from flask import Flask, render_template
from flask import Flask, render_template, request
import gensim.models.keyedvectors as word2vec
import KMeans
import json
import os
app= Flask(__name__)
@app.route("/", methods=['GET', 'POST'])
def home_page():
    # Nếu là POST (gửi file)
    if request.method == "POST":
         try:
            # Lấy file gửi lên
            contents = request.form.get("input")
            k = request.form.get("K_Model")
            nameModel = request.form.get("nameModel")
            if(k == "default"):
                k = int(0)
            else:
                k = int(k)
            if contents:
                print(type(k))
                result = KMeans.modelKMeans(contents,nameModel,int(k))
                return render_template("index.html", output_text=result, input_text=contents)
            else:
                return render_template('index.html', msg='Hãy chọn file để tải lên')
         except Exception as ex:
            print(ex)
            return render_template('index.html')
    else:
        return render_template('index.html')
# API
@app.route("/summary_text", methods=["POST"])
def top_ratings():
    input = request.json["original_text"]
    nameModel = request.json["nameModel"]
    k = request.json["k"]
    # result = KMeans(input)
    result = KMeans.modelKMeans(input,nameModel,int(k))
    return json.dumps(result)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)