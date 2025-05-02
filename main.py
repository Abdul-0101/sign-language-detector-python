from flask import Flask, render_template, send_from_directory, request

app = Flask(
    __name__,
    static_folder="static",    # <— your JS/.wasm go here
    static_url_path=""         # <— serve them at “/foo.js” not “/static/foo.js”
)

# This route serves your index.html from templates/
@app.route("/")
def index():
    return render_template("index.html")

# Fallback for any other paths: serve from static/
@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# Ensure .wasm files get the correct MIME type
@app.after_request
def add_wasm_header(response):
    if request.path.endswith(".wasm"):
        response.headers["Content-Type"] = "application/wasm"
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
