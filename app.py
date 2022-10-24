from flask import Flask, render_template
from flask import Flask, render_template, request
from sources.KMeans import KMeans
# call the Flask constructor
#application object

app= Flask(__name__)

#view function

@app.route("/", methods=['GET', 'POST'])
def index() :
    # Nếu là POST (gửi file)
    if request.method == "POST":
         try:
            # Lấy file gửi lên
            contents = request.form['input']
            print(contents)
            if contents:
                result = KMeans(contents)
                return render_template("index.html", output_text=result, input_text=contents)
            else:
                return render_template('index.html', msg='Hãy chọn file để tải lên')

         except Exception as ex:
            # Nếu lỗi thì thông báo
            print(ex)
            return render_template('index.html')

    else:
        # Nếu là GET thì hiển thị giao diện upload
        return render_template('index.html')

if __name__ == '__main__':
	
	app.run(debug=True)